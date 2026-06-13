from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException


def _make_mock_mv(
    version: str = "1",
    name: str = "test-model",
    run_id: str = "run-abc",
    tags: dict | None = None,
):
    """Фабрика MagicMock для объекта ModelVersion из mlflow."""
    mv = MagicMock()
    mv.name = name
    mv.version = version
    mv.run_id = run_id
    mv.source = f"models:/{name}/{version}"
    mv.creation_timestamp = 1_700_000_000_000
    mv.tags = tags or {}
    return mv


@pytest.fixture(autouse=True)
def env_defaults(monkeypatch):
    """Устанавливает окружение по умолчанию для всех тестов в файле."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", "test-model")
    monkeypatch.setenv("MLFLOW_MODEL_ALIAS", "production")


class TestEnvHelpers:

    def test_model_name_from_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", "my-rag")
        from src.mlflow_registry import _model_name

        assert _model_name() == "my-rag"

    def test_model_name_default(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_REGISTERED_MODEL_NAME", raising=False)
        from src.mlflow_registry import _model_name, DEFAULT_MODEL_NAME

        assert _model_name() == DEFAULT_MODEL_NAME

    def test_alias_from_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_MODEL_ALIAS", "staging")
        from src.mlflow_registry import _alias

        assert _alias() == "staging"

    def test_alias_default(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_MODEL_ALIAS", raising=False)
        from src.mlflow_registry import _alias, DEFAULT_ALIAS

        assert _alias() == DEFAULT_ALIAS


class TestModelVersionInfo:

    def test_all_fields_accessible(self):
        from src.mlflow_registry import ModelVersionInfo

        info = ModelVersionInfo(
            name="m",
            version="2",
            alias="production",
            run_id="r123",
            source="s3://bucket/model",
            creation_time=1700000000,
            tags={"env": "prod"},
        )
        assert info.name == "m"
        assert info.version == "2"
        assert info.alias == "production"
        assert info.run_id == "r123"
        assert info.tags == {"env": "prod"}

    def test_alias_can_be_none(self):
        from src.mlflow_registry import ModelVersionInfo

        info = ModelVersionInfo(
            name="m",
            version="1",
            alias=None,
            run_id=None,
            source=None,
            creation_time=None,
            tags={},
        )
        assert info.alias is None

    def test_is_dataclass_serializable(self):
        from src.mlflow_registry import ModelVersionInfo

        info = ModelVersionInfo(
            name="m",
            version="1",
            alias="production",
            run_id="r",
            source="s",
            creation_time=0,
            tags={},
        )
        d = asdict(info)
        assert isinstance(d, dict)
        assert d["name"] == "m"


class TestInfoToDict:

    def test_returns_none_for_none_input(self):
        from src.mlflow_registry import info_to_dict

        assert info_to_dict(None) is None

    def test_returns_dict_for_valid_info(self):
        from src.mlflow_registry import info_to_dict, ModelVersionInfo

        info = ModelVersionInfo(
            name="m",
            version="3",
            alias="production",
            run_id="r1",
            source="s",
            creation_time=123,
            tags={},
        )
        result = info_to_dict(info)
        assert isinstance(result, dict)
        assert result["version"] == "3"

    def test_dict_contains_all_fields(self):
        from src.mlflow_registry import info_to_dict, ModelVersionInfo

        info = ModelVersionInfo(
            name="rag",
            version="5",
            alias=None,
            run_id="abc",
            source="s3://x",
            creation_time=999,
            tags={"k": "v"},
        )
        result = info_to_dict(info)
        for field in (
            "name",
            "version",
            "alias",
            "run_id",
            "source",
            "creation_time",
            "tags",
        ):
            assert field in result


class TestGetActiveVersion:

    def test_returns_none_on_mlflow_exception(self):
        """Если alias не найден возвращает None, не поднимает исключение."""
        from src.mlflow_registry import get_active_version

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.get_model_version_by_alias.side_effect = MlflowException("not found")
            mock_client.return_value = client

            result = get_active_version()

        assert result is None

    def test_returns_model_version_info_on_success(self):
        from src.mlflow_registry import get_active_version, ModelVersionInfo

        mv = _make_mock_mv(version="4", run_id="run-xyz")

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.get_model_version_by_alias.return_value = mv
            mock_client.return_value = client

            result = get_active_version()

        assert isinstance(result, ModelVersionInfo)
        assert result.version == "4"
        assert result.run_id == "run-xyz"

    def test_alias_is_set_from_env(self):
        from src.mlflow_registry import get_active_version

        mv = _make_mock_mv(version="2")

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.get_model_version_by_alias.return_value = mv
            mock_client.return_value = client

            result = get_active_version()

        assert result.alias == "production"

    def test_tags_converted_to_plain_dict(self):
        from src.mlflow_registry import get_active_version

        mv = _make_mock_mv(version="1", tags={"stage": "prod", "team": "ml"})

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.get_model_version_by_alias.return_value = mv
            mock_client.return_value = client

            result = get_active_version()

        assert isinstance(result.tags, dict)
        assert result.tags["team"] == "ml"

    def test_queries_correct_model_name(self):
        from src.mlflow_registry import get_active_version

        mv = _make_mock_mv()

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.get_model_version_by_alias.return_value = mv
            mock_client.return_value = client

            get_active_version()

        client.get_model_version_by_alias.assert_called_once_with(
            name="test-model", alias="production"
        )


class TestListVersions:

    def test_returns_empty_list_on_search_exception(self):
        from src.mlflow_registry import list_versions

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.search_model_versions.side_effect = MlflowException("error")
            mock_client.return_value = client

            assert list_versions() == []

    def test_sorted_descending_by_version(self):
        from src.mlflow_registry import list_versions

        versions = [_make_mock_mv(v) for v in ["1", "3", "2"]]

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.search_model_versions.return_value = versions
            client.get_model_version_by_alias.side_effect = MlflowException("no alias")
            mock_client.return_value = client

            result = list_versions()

        assert [r.version for r in result] == ["3", "2", "1"]

    def test_active_alias_marked_on_correct_version(self):
        from src.mlflow_registry import list_versions

        v1 = _make_mock_mv("1")
        v2 = _make_mock_mv("2")
        active_mv = _make_mock_mv("2")

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.search_model_versions.return_value = [v1, v2]
            client.get_model_version_by_alias.return_value = active_mv
            mock_client.return_value = client

            result = list_versions()

        by_version = {r.version: r for r in result}
        assert by_version["2"].alias == "production"
        assert by_version["1"].alias is None

    def test_alias_none_when_alias_lookup_fails(self):
        from src.mlflow_registry import list_versions

        versions = [_make_mock_mv("1"), _make_mock_mv("2")]

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.search_model_versions.return_value = versions
            client.get_model_version_by_alias.side_effect = MlflowException("no alias")
            mock_client.return_value = client

            result = list_versions()

        assert all(r.alias is None for r in result)

    def test_limit_caps_results(self):
        from src.mlflow_registry import list_versions

        versions = [_make_mock_mv(str(i)) for i in range(1, 25)]  # 24 версии

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.search_model_versions.return_value = versions
            client.get_model_version_by_alias.side_effect = MlflowException("no alias")
            mock_client.return_value = client

            result = list_versions(limit=5)

        assert len(result) <= 5

    def test_returns_model_version_info_objects(self):
        from src.mlflow_registry import list_versions, ModelVersionInfo

        versions = [_make_mock_mv("1")]

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.search_model_versions.return_value = versions
            client.get_model_version_by_alias.side_effect = MlflowException("no alias")
            mock_client.return_value = client

            result = list_versions()

        assert len(result) == 1
        assert isinstance(result[0], ModelVersionInfo)

    def test_empty_registry_returns_empty_list(self):
        from src.mlflow_registry import list_versions

        with patch("src.mlflow_registry._client") as mock_client:
            client = MagicMock()
            client.search_model_versions.return_value = []
            client.get_model_version_by_alias.side_effect = MlflowException("no alias")
            mock_client.return_value = client

            assert list_versions() == []


# Интеграционный тест с полным моком MLflow
class TestRegisterNewVersion:

    @pytest.fixture
    def mock_mlflow_stack(self):
        """Патчит весь стек MLflow для register_new_version."""
        mock_version = _make_mock_mv(version="1")
        mock_run_info = MagicMock()
        mock_run_info.info.run_id = "run-new-123"

        run_ctx = MagicMock()
        run_ctx.__enter__ = MagicMock(return_value=mock_run_info)
        run_ctx.__exit__ = MagicMock(return_value=False)

        model_info = MagicMock()
        model_info.model_uri = "models:/test-model/1"

        client = MagicMock()
        client.search_model_versions.return_value = [mock_version]

        with patch("src.mlflow_registry._client", return_value=client), patch(
            "mlflow.start_run", return_value=run_ctx
        ), patch("mlflow.set_experiment"), patch("mlflow.set_tracking_uri"), patch(
            "mlflow.log_param"
        ), patch(
            "mlflow.pyfunc.log_model", return_value=model_info
        ):
            yield client, mock_version

    def test_returns_model_version_info(self, mock_mlflow_stack):
        from src.mlflow_registry import register_new_version, ModelVersionInfo

        _, _ = mock_mlflow_stack

        result = register_new_version(
            {"llm": "qwen2.5:7b", "k": 8}, promote_alias=False
        )

        assert isinstance(result, ModelVersionInfo)
        assert result.version == "1"

    def test_promote_alias_calls_set_alias(self, mock_mlflow_stack):
        from src.mlflow_registry import register_new_version

        client, _ = mock_mlflow_stack

        register_new_version({"llm": "qwen"}, promote_alias=True)

        client.set_registered_model_alias.assert_called_once_with(
            name="test-model", alias="production", version="1"
        )

    def test_no_promote_alias_skips_set_alias(self, mock_mlflow_stack):
        from src.mlflow_registry import register_new_version

        client, _ = mock_mlflow_stack

        register_new_version({"llm": "qwen"}, promote_alias=False)

        client.set_registered_model_alias.assert_not_called()

    def test_alias_none_when_not_promoted(self, mock_mlflow_stack):
        from src.mlflow_registry import register_new_version, ModelVersionInfo

        result = register_new_version({"llm": "qwen"}, promote_alias=False)
        assert isinstance(result, ModelVersionInfo)
        assert result.alias is None

    def test_description_calls_update_model_version(self, mock_mlflow_stack):
        from src.mlflow_registry import register_new_version

        client, _ = mock_mlflow_stack

        register_new_version({"k": 5}, description="v1 baseline", promote_alias=False)

        client.update_model_version.assert_called_once_with(
            name="test-model", version="1", description="v1 baseline"
        )

    def test_no_description_skips_update(self, mock_mlflow_stack):
        from src.mlflow_registry import register_new_version

        client, _ = mock_mlflow_stack

        register_new_version({"k": 5}, promote_alias=False)

        client.update_model_version.assert_not_called()
