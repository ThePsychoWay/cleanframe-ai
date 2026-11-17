"""run_local.py --lan: what listens where, and what each part is told. The script is loaded as a module; nothing is started."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "run_local.py"


@pytest.fixture(scope="module")
def run_local():  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("run_local_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def test_the_normal_plan_is_unchanged_and_not_open_to_the_network(run_local) -> None:  # noqa: ANN001
    plan = run_local.make_plan(8000, 3000, "192.168.1.20", lan=False)
    assert plan.phone_url is None and plan.web_host is None
    assert plan.build_env["NEXT_PUBLIC_API_URL"] == "http://localhost:8000"
    assert plan.api_env == {"API_PUBLIC_URL": "http://localhost:8000", "CORS_ORIGINS": "http://localhost:3000"}
    assert "LOCAL_LAN" not in plan.api_env


def test_the_lan_plan_opens_only_the_web_app_and_forwards_the_api_through_it(run_local) -> None:  # noqa: ANN001
    plan = run_local.make_plan(8000, 3000, "192.168.1.20", lan=True, hostname="my-pc")
    assert plan.phone_url == "http://192.168.1.20:3000"
    assert plan.web_host == "0.0.0.0"
    assert plan.api_origin == "http://127.0.0.1:8000"  # the API itself is never given a network address
    assert plan.build_env == {"NEXT_PUBLIC_API_URL": "same-origin", "API_PROXY_TARGET": "http://127.0.0.1:8000"}
    assert plan.api_env["LOCAL_LAN"] == "true" and plan.api_env["API_PUBLIC_URL"] == ""
    origins = plan.api_env["CORS_ORIGINS"].split(",")
    assert "http://192.168.1.20:3000" in origins and "http://localhost:3000" in origins and "http://my-pc:3000" in origins
    assert all(o.endswith(":3000") for o in origins)  # only the web app's port, never the API's


def test_switching_modes_forces_a_rebuild_because_the_api_address_is_built_in(run_local) -> None:  # noqa: ANN001
    normal = run_local.make_plan(8000, 3000, None, lan=False)
    phone = run_local.make_plan(8000, 3000, "192.168.1.20", lan=True)
    assert normal.build_api != phone.build_api
    # ...but a changing home-network address (DHCP) must NOT force one: it is only given to the API at start-up.
    assert phone.build_api == run_local.make_plan(8000, 3000, "10.0.0.7", lan=True).build_api


def test_lan_without_a_network_is_an_error_not_a_guess(run_local) -> None:  # noqa: ANN001
    with pytest.raises(ValueError):
        run_local.make_plan(8000, 3000, None, lan=True)


def test_the_address_found_is_a_real_network_address(run_local) -> None:  # noqa: ANN001
    found = run_local.lan_address()
    assert found is None or (found.count(".") == 3 and not found.startswith(("127.", "0.")))
