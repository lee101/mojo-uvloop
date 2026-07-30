import asyncio
import contextvars
import inspect
import threading
import warnings

import pytest
import uvloop

import mojo_uvloop as muv


def _observe_callbacks(factory):
    loop = factory()
    events = []
    cancelled = None

    def schedule_timers():
        loop.call_later(0.002, events.append, "later-2")
        loop.call_later(0.001, events.append, "later-1")
        loop.call_later(0.006, loop.stop)

    def schedule():
        nonlocal cancelled
        cancelled = loop.call_soon(events.append, "cancelled")
        cancelled.cancel()
        loop.call_soon(events.append, "soon-1")
        loop.call_later(-1, events.append, "soon-2")
        loop.call_soon(schedule_timers)

    try:
        loop.call_soon(schedule)
        loop.run_forever()
        assert cancelled is not None
        return events, cancelled.cancelled()
    finally:
        loop.close()


def test_callback_order_and_cancellation_match_uvloop():
    assert _observe_callbacks(muv.new_event_loop) == _observe_callbacks(
        uvloop.new_event_loop
    )


def test_call_soon_context_matches_uvloop():
    marker = contextvars.ContextVar("marker", default="default")

    def observe(factory):
        loop = factory()
        values = []
        context = contextvars.copy_context()
        context.run(marker.set, "captured")
        try:
            loop.call_soon(lambda: values.append(marker.get()), context=context)
            loop.call_soon(loop.stop)
            loop.run_forever()
            return values
        finally:
            loop.close()

    assert observe(muv.new_event_loop) == observe(uvloop.new_event_loop) == ["captured"]


def test_call_soon_threadsafe_executes_from_another_thread():
    loop = muv.new_event_loop()
    values = []
    loop_running = threading.Event()

    def schedule_from_thread():
        assert loop_running.wait(timeout=2)
        loop.call_soon_threadsafe(values.append, "thread")
        loop.call_soon_threadsafe(loop.stop)

    thread = threading.Thread(target=schedule_from_thread)
    try:
        thread.start()
        loop.call_soon(loop_running.set)
        loop.run_forever()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert values == ["thread"]
    finally:
        loop.close()


@pytest.mark.parametrize(
    ("delay", "immediate"),
    [(-1.0, True), (-float("inf"), True), (0.0004, True), (0.0005, True), (0.0015, False)],
)
def test_timer_quantization_handle_kind_matches_uvloop(delay, immediate):
    def is_immediate(factory):
        loop = factory()
        try:
            handle = loop.call_later(delay, lambda: None)
            result = not hasattr(handle, "when")
            handle.cancel()
            return result
        finally:
            loop.close()

    assert is_immediate(muv.new_event_loop) == is_immediate(uvloop.new_event_loop)
    assert is_immediate(muv.new_event_loop) is immediate


def test_nan_timer_error_matches_uvloop():
    for factory in (muv.new_event_loop, uvloop.new_event_loop):
        loop = factory()
        try:
            with pytest.raises(ValueError, match="NaN"):
                loop.call_later(float("nan"), lambda: None)
        finally:
            loop.close()


def test_call_at_executes_and_is_cancellable():
    loop = muv.new_event_loop()
    values = []
    try:
        cancelled = loop.call_at(loop.time(), values.append, "cancelled")
        cancelled.cancel()
        loop.call_at(loop.time() + 0.001, values.append, "called")
        loop.call_later(0.004, loop.stop)
        loop.run_forever()
        assert values == ["called"]
    finally:
        loop.close()


def test_run_signature_and_result_match_uvloop():
    assert tuple(inspect.signature(muv.run).parameters) == tuple(
        inspect.signature(uvloop.run).parameters
    )

    async def identify():
        loop = asyncio.get_running_loop()
        await asyncio.sleep(0)
        return type(loop).__name__, 42

    assert muv.run(identify()) == uvloop.run(identify()) == ("Loop", 42)


def test_run_rejects_wrong_loop_factory_like_uvloop():
    async def nothing():
        return None

    with pytest.raises(TypeError, match="non-mojo-uvloop"):
        muv.run(nothing(), loop_factory=asyncio.SelectorEventLoop)


def test_task_future_and_executor_behavior_matches_uvloop():
    async def exercise():
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        loop.call_soon(future.set_result, 20)
        task = loop.create_task(asyncio.sleep(0, result=22), name="answer")
        threaded = await loop.run_in_executor(None, lambda: 1)
        return await future + await task, task.get_name(), threaded

    assert muv.run(exercise()) == uvloop.run(exercise()) == (42, "answer", 1)


def test_tcp_echo_behavior_matches_uvloop():
    async def exercise():
        async def echo(reader, writer):
            data = await reader.readexactly(12)
            writer.write(data[::-1])
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"event-loops!")
        await writer.drain()
        response = await reader.readexactly(12)
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()
        return response

    assert muv.run(exercise()) == uvloop.run(exercise()) == b"!spool-tneve"


def test_bulk_call_soon_and_later_execute_all_callbacks():
    loop = muv.new_event_loop()
    events = []
    try:
        loop.call_soon_many(
            [events.append, events.append], args=[("soon-a",), ("soon-b",)]
        )
        loop.call_later_many(
            [0.003, 0.001, 0.002],
            [events.append] * 3,
            args=[("later-c",), ("later-a",), ("later-b",)],
        )
        loop.call_later(0.008, loop.stop)
        loop.run_forever()
        assert events == ["soon-a", "soon-b", "later-a", "later-b", "later-c"]
    finally:
        loop.close()


def test_call_at_many_and_argument_validation():
    loop = muv.new_event_loop()
    events = []
    try:
        now = loop.time()
        loop.call_at_many(
            [now + 0.002, now + 0.001],
            [events.append, events.append],
            args=[("b",), ("a",)],
        )
        loop.call_later(0.006, loop.stop)
        loop.run_forever()
        assert events == ["a", "b"]
        with pytest.raises(ValueError, match="equal length"):
            loop.call_later_many([1, 2], [lambda: None])
    finally:
        loop.close()


def test_policy_factory_and_validation_match_uvloop():
    policy = muv.EventLoopPolicy()
    loop = policy.new_event_loop()
    try:
        policy.set_event_loop(loop)
        assert policy.get_event_loop() is loop
        with pytest.raises(TypeError, match="AbstractEventLoop"):
            policy.set_event_loop(object())
    finally:
        policy.set_event_loop(None)
        loop.close()
    with pytest.raises(RuntimeError, match="no current event loop"):
        policy.get_event_loop()


def test_install_sets_policy_and_warns():
    old_policy = asyncio.get_event_loop_policy()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            muv.install()
        assert isinstance(asyncio.get_event_loop_policy(), muv.EventLoopPolicy)
        assert any(item.category is DeprecationWarning for item in caught)
    finally:
        asyncio.set_event_loop_policy(old_policy)
