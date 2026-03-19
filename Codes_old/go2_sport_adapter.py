import time
import math
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient


class Go2SportAdapter:
    """
    - Subscribes to: rt/sportmodestate (position, velocity, imu rpy, etc.)  :contentReference[oaicite:2]{index=2}
    - Sends commands via: SportClient.Move(vx, vy, vyaw)                  :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, net_ifname: str | None = None, timeout_s: float = 10.0):
        # DDS init (domain 0). Many examples pass interface name (e.g., enp3s0/eth0). :contentReference[oaicite:4]{index=4}
        if net_ifname is None or net_ifname == "":
            ChannelFactoryInitialize(0)
        else:
            ChannelFactoryInitialize(0, net_ifname)

        self.client = SportClient()
        self.client.SetTimeout(timeout_s)
        self.client.Init()

        self._last_state: SportModeState_ | None = None

        # High-level motion state subscription. Examples commonly use "rt/sportmodestate". :contentReference[oaicite:5]{index=5}
        self.sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sub.Init(self._state_cb, 10)

    def _state_cb(self, msg: SportModeState_):
        self._last_state = msg

    def wait_for_state(self, timeout_s: float = 5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self._last_state is not None:
                return True
            time.sleep(0.01)
        return False

    def get_xy_yaw(self):
        """
        SportModeState includes position[3] and imu_state.rpy[3] in ROS2 docs and python examples. :contentReference[oaicite:6]{index=6}
        """
        if self._last_state is None:
            return None
        x = float(self._last_state.position[0])
        y = float(self._last_state.position[1])
        yaw = float(self._last_state.imu_state.rpy[2])
        return x, y, yaw

    @staticmethod
    def world_to_body(vx_w: float, vy_w: float, yaw: float):
        # v_body = R(-yaw) v_world
        c, s = math.cos(yaw), math.sin(yaw)
        vx_b =  c * vx_w + s * vy_w
        vy_b = -s * vx_w + c * vy_w
        return vx_b, vy_b

    def stand_up(self):
        return self.client.StandUp()

    def stop_move(self):
        return self.client.StopMove()

    def damp(self):
        return self.client.Damp()

    def send_move_body(self, vx_b: float, vy_b: float, vyaw: float):
        # Clip to Sport mode ranges (from Unitree docs). :contentReference[oaicite:7]{index=7}
        vx_b = float(np.clip(vx_b, -2.5, 3.8))
        vy_b = float(np.clip(vy_b, -1.0, 1.0))
        vyaw = float(np.clip(vyaw, -4.0, 4.0))
        return self.client.Move(vx_b, vy_b, vyaw)

    def send_move_world(self, vx_w: float, vy_w: float, vyaw: float = 0.0):
        st = self.get_xy_yaw()
        if st is None:
            return None
        _, _, yaw = st
        vx_b, vy_b = self.world_to_body(vx_w, vy_w, yaw)
        return self.send_move_body(vx_b, vy_b, vyaw)
