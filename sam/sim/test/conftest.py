import pytest

pytest.register_assert_rewrite("sam.sim.test.test.check_arr")


def pytest_addoption(parser):
    parser.addoption("--debug-sim", action="store_true", default=False,
                     help="Emit debug print statements. Use with `-s`")
    parser.addoption("--ssname", action="store", help="Suitesparse name for the end-to-end test")
    parser.addoption("--report-stats", action="store_true", default=False,
                     help="Flag that enables statistics reporting")
    parser.addoption("--frosttname", action="store", help="Frostt name for the end-to-end test")
    parser.addoption("--vecname", action="store", help="Vector name for the end-to-end test")
    parser.addoption("--check-gold", action="store_true", default=False,
                     help="Flag that enables functional output checking")
    parser.addoption("--result-out", action="store",
                     help="Store output to filename for functional output checking")
    parser.addoption("--synth", action="store_true", default=False,
                     help="Flag that enables functional output checking")
    parser.addoption("--skip-empty", action="store_true", default=False,
                     help="Flag that enables functional output checking")
    parser.addoption("--yaml-name", type=str, default="memory_config.yaml",
                     help="Name of yaml file for tiling memory configuration")
    parser.addoption("--nbuffer", action="store_true", default=False,
                     help="If nbuffering is enabled")
    parser.addoption("--back", action="store_true", default=False,
                     help="Whether backpressure is enabled")
    parser.addoption("--memory-model", action="store_true", default=False,
                     help="Whether memory model is wanted")
    parser.addoption("--depth", action="store", default=2,
                     help="fifo depth value")
    parser.addoption("--nnz-value", action="store", default=5000,
                     help="nnz value for stats")
    parser.addoption("--cast", action="store_true", default=False,
                     help="Flag that runs all simulations using integer input "
                          "and output data (used for hardware simulation comparison, "
                          "when data is formatted with the '-cast' flag)")
    parser.addoption("--positive-only", action="store_true", default=False,
                     help="Flag that casts all inputs to positive values only "
                          "(used for hardware simulation comparison)")


def pytest_configure(config):
    config.addinivalue_line("markers", "suitesparse: mark test as needing suitesparse dataset to run")
    config.addinivalue_line("markers", "frostt: mark test as needing suitesparse dataset to run")
    config.addinivalue_line("markers", "vec: mark test as needing suitesparse dataset to run")
    config.addinivalue_line("markers", "synth: mark test as needing synthetic dataset to run")


def pytest_collection_modifyitems(config, items):
    skip_ss = pytest.mark.skip(reason="Need --ssname option to run")
    skip_frostt = pytest.mark.skip(reason="Need --frosttname option to run")
    skip_vec = pytest.mark.skip(reason="Need --vecname option to run")
    skip_synth = pytest.mark.skip(reason="Need --synth option to run")
    if not config.getoption("--ssname"):
        # --ssname given in cli: do not skip apps/ tests
        for item in items:
            if "suitesparse" in item.keywords:
                item.add_marker(skip_ss)
    if not config.getoption("--frosttname"):
        for item in items:
            if "frostt" in item.keywords:
                item.add_marker(skip_frostt)

    if not config.getoption("--vecname"):
        for item in items:
            if "vec" in item.keywords:
                item.add_marker(skip_vec)
    if not config.getoption("--synth"):
        # --synth given in cli: do not skip apps/ tests
        for item in items:
            if "synth" in item.keywords:
                item.add_marker(skip_synth)


@pytest.fixture
def backpressure(request):
    return request.config.getoption("--back")


@pytest.fixture
def memory_model(request):
    return request.config.getoption("--memory-model")


@pytest.fixture
def skip_empty(request):
    return request.config.getoption("--skip-empty")


@pytest.fixture
def nbuffer(request):
    return request.config.getoption("--nbuffer")


@pytest.fixture
def debug_sim(request):
    return request.config.getoption("--debug-sim")


@pytest.fixture
def depth(request):
    return request.config.getoption("--depth")


@pytest.fixture
def nnz_value(request):
    return request.config.getoption("--nnz-value")


@pytest.fixture
def check_gold(request):
    return request.config.getoption("--check-gold")


@pytest.fixture
def report_stats(request):
    return request.config.getoption("--report-stats")


@pytest.fixture
def ssname(request):
    return request.config.getoption("--ssname")


@pytest.fixture
def result_out(request):
    return request.config.getoption("--result-out")


@pytest.fixture
def frosttname(request):
    return request.config.getoption("--frosttname")


@pytest.fixture
def vecname(request):
    return request.config.getoption("--vecname")


@pytest.fixture
def synth(request):
    return request.config.getoption("--synth")


@pytest.fixture
def yaml_name(request):
    return request.config.getoption("--yaml-name")


@pytest.fixture
def cast(request):
    return request.config.getoption("--cast")


@pytest.fixture
def positive_only(request):
    return request.config.getoption("--positive-only")


@pytest.fixture
def samBench(benchmark):
    def f(func, extra_info=None, save_ret_val=False):
        # Take statistics based on 10 rounds.
        if extra_info is not None:
            for k, v in extra_info.items():
                benchmark.extra_info[k] = v
        if save_ret_val:
            benchmark.extra_info["return"] = func()
        benchmark.pedantic(func, rounds=1, iterations=1, warmup_rounds=0)

    return f
