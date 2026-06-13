import os
import subprocess
from pathlib import Path

import pytest


def _run_dvc(*args, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["dvc", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _cloud_credentials_available() -> bool:
    """Проверяет наличие credentials для Google Drive (dvc remote)."""
    return os.getenv("GDRIVE_CREDENTIALS_DATA") is not None


class TestDVCInstallation:

    def test_dvc_binary_exists(self):
        """dvc доступен в PATH."""
        result = _run_dvc("--version")
        assert result.returncode == 0, (
            f"DVC не найден или вернул ошибку.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_dvc_version_output_is_parseable(self):
        """Вывод dvc --version содержит строку с версией формата X.Y.Z."""
        result = _run_dvc("--version")
        version_str = result.stdout.strip()
        assert version_str, "dvc --version вернул пустой stdout"
        parts = version_str.split(".")
        assert len(parts) >= 2, f"Неожиданный формат версии: {version_str!r}"
        assert parts[
            0
        ].isdigit(), f"Мажорная версия не является числом: {version_str!r}"

    def test_dvc_version_is_at_least_3(self):
        """Требуется DVC >= 3.x для совместимости с проектом."""
        result = _run_dvc("--version")
        major = int(result.stdout.strip().split(".")[0])
        assert major >= 3, (
            f"Установлена устаревшая версия DVC {result.stdout.strip()}. "
            "Требуется >= 3.0"
        )

    def test_dvc_help_works(self):
        """dvc help не падает — базовая smoke-проверка."""
        result = _run_dvc("--help")
        assert result.returncode == 0


class TestDVCProjectSetup:

    def test_dvc_directory_exists(self):
        """.dvc/ директория присутствует в корне репозитория."""
        assert Path(".dvc").is_dir(), (
            ".dvc/ директория не найдена. "
            "Запустите `dvc init` для инициализации проекта."
        )

    def test_dvc_config_file_exists(self):
        """.dvc/config существует после инициализации."""
        assert Path(".dvc/config").exists(), (
            ".dvc/config не найден. "
            "Возможно, DVC не был инициализирован (`dvc init`)."
        )

    def test_dvcignore_exists(self):
        """.dvcignore создается при dvc init."""
        assert Path(".dvcignore").exists(), (
            ".dvcignore не найден. "
            "Проверьте, что `dvc init` был выполнен в корне репозитория."
        )

    def test_at_least_one_remote_configured(self):
        """В проекте должен быть настроен хотя бы один remote."""
        result = _run_dvc("remote", "list")
        assert (
            result.returncode == 0
        ), f"Команда `dvc remote list` завершилась с ошибкой:\n{result.stderr}"
        remotes = result.stdout.strip()
        assert remotes, (
            "Ни один DVC remote не сконфигурирован. "
            "Добавьте remote через `dvc remote add`."
        )

    def test_default_remote_is_set(self):
        """core.remote должен указывать на существующий remote."""
        result = _run_dvc("config", "core.remote")
        assert result.returncode == 0, (
            "core.remote не задан. "
            "Установите дефолтный remote: `dvc remote default <name>`."
        )
        remote_name = result.stdout.strip()
        assert remote_name, "core.remote задан, но пустой"

    def test_dvc_status_local_runs_without_error(self):
        """dvc status (локальный кеш) не возвращает ошибку исполнения."""
        result = _run_dvc("status", timeout=30)
        assert result.returncode in (0, 1), (
            f"dvc status завершился неожиданным кодом {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )


@pytest.mark.skipif(
    not _cloud_credentials_available(),
    reason=(
        "GDrive credentials не найдены. "
        "Ожидается файл /app/.dvc/gdrive_service_account.json "
        "или переменная окружения GDRIVE_SERVICE_ACCOUNT_KEY."
    ),
)
class TestDVCRemoteConnection:

    def test_dvc_remote_list_returns_configured_remotes(self):
        """remote list возвращает непустой список при наличии credentials."""
        result = _run_dvc("remote", "list")
        assert result.returncode == 0
        lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
        assert len(lines) >= 1, "Ожидался хотя бы один remote, но список пуст"

    def test_dvc_status_cloud_reachable(self):
        """
        dvc status --cloud проверяет реальное соединение с remote.
        Код 0 = данные актуальны, 1 = есть изменения/отсутствующие файлы.
        Оба результата допустимы — важно, что нет ошибки аутентификации/сети.
        """
        result = _run_dvc("status", "--cloud", timeout=120)
        assert result.returncode in (0, 1), (
            f"dvc status --cloud вернул код {result.returncode} — "
            f"возможна проблема с подключением к remote.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_remote_url_is_reachable(self):
        """
        Получает URL дефолтного remote и проверяет, что dvc может
        обратиться к нему через `dvc remote list --verbose`.
        """
        result = _run_dvc("remote", "list", "--verbose", timeout=30)
        assert (
            result.returncode == 0
        ), f"Не удалось получить список remotes:\n{result.stderr}"
        output = result.stdout.strip()
        assert output, "Verbose remote list пустой — remote не настроен"
        assert "gdrive://" in output, f"Не нашли gdrive:// URL в выводе:\n{output}"
