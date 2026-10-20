"""Open the saved FANUC workcell when Isaac's streaming app starts."""

import asyncio

import omni.kit.app
import omni.kit.viewport.utility
import omni.usd


STAGE_PATH = "/workspace/team-data/fanuc_cell.usd"
CAMERA_PATH = "/World/TableCam"


async def open_workcell() -> None:
    context = omni.usd.get_context()
    await context.open_stage_async(STAGE_PATH)

    # Let the streamed viewport observe the new stage before switching cameras.
    app = omni.kit.app.get_app()
    for _ in range(3):
        await app.next_update_async()

    viewport = omni.kit.viewport.utility.get_active_viewport()
    if viewport is not None:
        viewport.set_active_camera(CAMERA_PATH)


asyncio.ensure_future(open_workcell())
