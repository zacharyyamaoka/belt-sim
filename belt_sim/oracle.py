"""The facility model's F1 line_sim, restated as a scenario of this engine (the independent oracle).

F1 (bam_facility_model/line_sim.py) is event-driven and constant-speed: arm k owns the window [k*w, (k+1)*w), a
free arm commits to the most downstream item already on the belt that will still be in its window ``reach_time_s``
later, and is busy for a fixed ``cycle_time_s``; the grasp outcome is drawn at contact and a miss releases the item.
Same rules here: rectangular windows, the fixed timing model, the MostDownstream picker and a Poisson stream with no
spacing. Agreement on recovery says the fixed-step engine moves, gates and books items the way an exact event loop
does; it is the check that can fail.
"""

from __future__ import annotations

from .config import ArmSpec, BeltSpec, Bin, Category, Scenario, Supply


def line_sim_scenario(items_per_s, belt_speed_m_s, arms, window_m, cycle_time_s, reach_time_s, grasp_success):
    arm_specs = tuple(
        ArmSpec(f"arm{k + 1}", (k + 0.5) * window_m, 0.3, (Bin("c0", (k + 0.5) * window_m, -0.15),),
                fixed_cycle_s=cycle_time_s, fixed_reach_s=reach_time_s, window_x=(k * window_m, (k + 1) * window_m),
                v_track_max_m_s=10.0)
        for k in range(arms)
    )
    belt = BeltSpec(length_m=arms * window_m, v_max_m_s=belt_speed_m_s, v_min_m_s=0.0, accel_m_s2=1e9)
    return Scenario("F1 line_sim", belt, arm_specs, (Category("c0", 1.0, 1.0, grasp_success),),
                    Supply(mode="stream", rate_per_s=items_per_s, min_gap_m=0.0))
