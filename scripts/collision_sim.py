import pickle
import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

MODEL_XML   = "../flymimic/assets/models/nmf_collision_v2.xml"
ANGLE_FILE  = "../logs/leg_joint_angles_0-750_1e4Hz_6legs.pkl"
MUSCLE_FILE = "../logs/model_sda_no_play_cvt.npz"   # LF muscle activation file

# PD gains for joint-space tracking (non-LF joints)
KP = 800.0
KD = 5.0

# END_FRAME is EXCLUSIVE: frames [START_FRAME, END_FRAME)
START_FRAME = 10000
END_FRAME   = 14200
NUM_CYCLES  = 10
FRAMES_PER_ACT = 20  # dt_muscle/dt_sim = 0.002 / 0.0001 = 20
SMOOTH_WINDOW_STEPS = 100
SAVGOL_WINDOW = 301
SAVGOL_POLY   = 3

# -------------------------------------------------------
# Load angle data
# -------------------------------------------------------
with open(ANGLE_FILE, "rb") as f:
    angle_data = pickle.load(f)  # dict: key -> np.array(T,)
angle_data['Angle_RM_FTi_pitch'] += 0.2

N_STEPS = len(next(iter(angle_data.values())))
assert 0 <= START_FRAME < END_FRAME <= N_STEPS, "Frame range out of bounds"

segment_len = END_FRAME - START_FRAME   # number of frames in the segment (expected 4200)
print(f"Segment length (frames): {segment_len}")

# -------------------------------------------------------
# LF muscle names (order must match actuator order & npz keys)
# -------------------------------------------------------
lf_muscle_names = [
    "LFC_tergopleural_promotor_a",
    "LFC_tergopleural_promotor_b",
    "LFC_pleural_remotor_and_abductor",
    "LFC_pleural_promotor",
    "LFC_sternal_anterior_rotator",
    "LFC_sternal_posterior_rotator",
    "LFC_sternal_adductor",
    "LFF_trochanter_flexor_b",
    "LFF_sterno-tergo-trochanter_extensor_a",
    "LFF_sterno-tergo-trochanter_extensor_b",
    "LFF_accesory_trochanter_flexor",
    "LFF_trochanter_extensor",
    "LFF_trochanter_flexor_a",
    "LFTibia_flex_93434",
    "LFTibia_extensor_93932",
]

lf_tendon_names = [mn + '_tendon' for mn in lf_muscle_names]
npz = np.load(MUSCLE_FILE)

# Confirm all keys exist
missing = [name for name in lf_muscle_names if name not in npz.files]
if missing:
    raise KeyError(f"Missing muscle keys in {MUSCLE_FILE}: {missing}")

# Stack each (210,) activation into a (210, 15) array
muscle_raw_list = []
for name in lf_muscle_names:
    arr = np.asarray(npz[name])  # shape (210,)
    muscle_raw_list.append(arr)

muscle_raw = np.stack(muscle_raw_list, axis=1)  # shape (T_muscle, N_muscle) = (210, 15)
T_muscle, N_muscle = muscle_raw.shape

assert N_muscle == 15, f"Expected 15 LF muscles, got {N_muscle}"
assert T_muscle == 210, f"Expected T=210 in {MUSCLE_FILE}, got {T_muscle}"
assert segment_len == T_muscle * FRAMES_PER_ACT, (
    f"segment_len={segment_len} but T_muscle*FRAMES_PER_ACT={T_muscle * FRAMES_PER_ACT}"
)

# -------------------------------------------------------
# Helper: build joint index from model
# -------------------------------------------------------
def build_joint_index(model):
    """
    Returns:
      joint_info: dict name -> dict with qpos_idx, qvel_idx, limited(bool), range(np.array(2,))
    """
    joint_info = {}

    for j_id in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j_id)
        if name is None:
            continue  # free joint has no name in this model
        qpos_idx = model.jnt_qposadr[j_id]
        qvel_idx = model.jnt_dofadr[j_id]
        limited = bool(model.jnt_limited[j_id])
        rng = np.array(model.jnt_range[2 * j_id : 2 * j_id + 2])
        joint_info[name] = {
            "qpos_idx": qpos_idx,
            "qvel_idx": qvel_idx,
            "limited": limited,
            "range": rng,
        }
    return joint_info

# -------------------------------------------------------
# Mapping: angle-file → MuJoCo joint names
# (LF is muscle-only, so we do NOT use LF here)
# -------------------------------------------------------
FRONT_CONFIG_NONLF = {
    "RF": dict(
        coxa_prefix="joint_RFCoxa",
        troch_prefix="joint_RFTrochanter",
        tibia_joint="joint_RFTibia_pitch",
    ),
}

MID_HIND_CONFIG = {
    "LM": dict(
        coxa_prefix="joint_LMCoxa",
        femur_prefix="joint_LMFemur",
        tibia_joint="joint_LMTibia",
    ),
    "LH": dict(
        coxa_prefix="joint_LHCoxa",
        femur_prefix="joint_LHFemur",
        tibia_joint="joint_LHTibia",
    ),
    "RM": dict(
        coxa_prefix="joint_RMCoxa",
        femur_prefix="joint_RMFemur",
        tibia_joint="joint_RMTibia",
    ),
    "RH": dict(
        coxa_prefix="joint_RHCoxa",
        femur_prefix="joint_RHFemur",
        tibia_joint="joint_RHTibia",
    ),
}

LF_CONFIG = dict(
    coxa_prefix="joint_LFCoxa",
    troch_prefix="joint_LFTrochanter",
    tibia_joint="joint_LFTibia_pitch",
)


def compute_joint_targets_from_file_nonlf(i):
    """
    For time index i, read the angle file and return:
        joint_targets: dict { mujoco_joint_name: target_angle }
    for RF + LM + LH + RM + RH only.
    """
    jt = {}

    # ----- Front RIGHT leg only (RF) -----
    for leg, cfg in FRONT_CONFIG_NONLF.items():
        pref = f"Angle_{leg}_"

        jt[cfg["coxa_prefix"] + "_yaw"]   = angle_data[pref + "ThC_yaw"][i]
        jt[cfg["coxa_prefix"] + "_pitch"] = angle_data[pref + "ThC_pitch"][i]
        jt[cfg["coxa_prefix"] + "_roll"]  = angle_data[pref + "ThC_roll"][i]

        jt[cfg["troch_prefix"] + "_yaw"]   = angle_data[pref + "CTr_yaw"][i]
        jt[cfg["troch_prefix"] + "_pitch"] = angle_data[pref + "CTr_pitch"][i]
        jt[cfg["troch_prefix"] + "_roll"]  = angle_data[pref + "CTr_roll"][i]

        fti  = angle_data[pref + "FTi_pitch"][i]
        tita = angle_data[pref + "TiTa_pitch"][i]
        jt[cfg["tibia_joint"]] = fti + tita

    # ----- Mid & hind legs -----
    for leg, cfg in MID_HIND_CONFIG.items():
        pref = f"Angle_{leg}_"

        jt[cfg["coxa_prefix"] + "_yaw"]  = angle_data[pref + "ThC_yaw"][i]
        jt[cfg["coxa_prefix"]]           = angle_data[pref + "ThC_pitch"][i]
        jt[cfg["coxa_prefix"] + "_roll"] = angle_data[pref + "ThC_roll"][i]

        jt[cfg["femur_prefix"]]           = angle_data[pref + "CTr_pitch"][i]
        jt[cfg["femur_prefix"] + "_roll"] = angle_data[pref + "CTr_roll"][i]

        fti  = angle_data[pref + "FTi_pitch"][i]
        tita = angle_data[pref + "TiTa_pitch"][i]
        jt[cfg["tibia_joint"]] = fti + tita

    return jt


def clip_targets_to_ranges(joint_targets, joint_info):
    """
    Clip targets to the joint's range if the joint is limited.
    """
    clipped = {}
    for name, q_des in joint_targets.items():
        info = joint_info.get(name)
        if info is None:
            continue
        if info["limited"]:
            lo, hi = info["range"]
            if lo < hi:
                q_des = np.clip(q_des, lo, hi)
        clipped[name] = q_des
    return clipped

# -------------------------------------------------------
# Tendon color update
# -------------------------------------------------------
def build_lf_tendon_ids(model, tendon_names):
    ids = []
    for name in tendon_names:
        tid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_TENDON, name)
        if tid < 0:
            raise ValueError(f"Tendon '{name}' not found in model.")
        ids.append(tid)
    return np.array(ids, dtype=int)


def update_tendon_colors(model, lf_tendon_ids, act_vec):
    act = np.clip(act_vec, 0.0, 1.0)
    for idx, tid in enumerate(lf_tendon_ids):
        a = float(act[idx])
        r = a
        g = 0.0
        b = 1.0 - a
        model.tendon_rgba[4*tid : 4*tid + 4] = np.array([r, g, b, 1.0], dtype=float)

# -------------------------------------------------------
# Contact force logging helpers
# -------------------------------------------------------
def build_leg_tarsus_geom_ids(model):
    """
    Get geom ids for all tarsus segments (1..5) of each leg, and the floor.
    Assumes geom names:
        LFTarsus1_geom..LFTarsus5_geom, etc.
    """
    legs = ["LF", "RF", "LM", "RM", "LH", "RH"]

    tarsus_geom_names = {}
    for leg in legs:
        tarsus_geom_names[leg] = [f"{leg}Tarsus{i}_geom" for i in range(1, 6)]

    leg_geom_ids = {}
    for leg, names in tarsus_geom_names.items():
        ids = []
        for gname in names:
            gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, gname)
            if gid < 0:
                raise ValueError(f"Geom '{gname}' for leg '{leg}' not found in model.")
            ids.append(gid)
        leg_geom_ids[leg] = ids

    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("Geom 'floor' not found in model.")

    geomid_to_leg = {}
    for leg, ids in leg_geom_ids.items():
        for gid in ids:
            geomid_to_leg[gid] = leg

    return floor_id, leg_geom_ids, geomid_to_leg

def smooth_signal(x, window):
    """
    Simple moving-average smoothing.
    x: 1D numpy array
    window: integer window size in samples
    """
    if window <= 1 or x.size == 0:
        return x
    window = min(window, x.size)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")

def save_contact_force_plot(time_log, grf_logs, out_path="contact_forces_6legs.png"):
    """
    Save a plot of smoothed summed contact-force magnitudes vs time for each leg.
    First apply moving average, then Savitzky–Golay filter.
    """
    if len(time_log) == 0:
        print("No contact data recorded; skipping GRF plot.")
        return

    time_arr = np.asarray(time_log)
    legs = ["LF", "RF", "LM", "RM", "LH", "RH"]
    fig, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axs = axs.ravel()

    for i, leg in enumerate(legs):
        ax = axs[i]
        forces = np.asarray(grf_logs.get(leg, []), dtype=float)
        if forces.size == 0:
            continue

        # 1) Moving-average smoothing
        forces_ma = smooth_signal(forces, SMOOTH_WINDOW_STEPS)

        # 2) Savitzky–Golay smoothing on top
        n = forces_ma.size
        if n > 5:
            win = min(SAVGOL_WINDOW, n)
            # window length must be odd and > polyorder
            if win % 2 == 0:
                win -= 1
            if win <= SAVGOL_POLY:
                forces_smooth = forces_ma
            else:
                forces_smooth = savgol_filter(forces_ma, window_length=win, polyorder=SAVGOL_POLY)
        else:
            forces_smooth = forces_ma

        ax.plot(time_arr, forces_smooth)
        ax.set_title(f"{leg} contact force (smoothed + SavGol)")
        ax.set_ylabel("|F_contact| [μN]")
        ax.grid(True, linestyle="--", alpha=0.3)

    for ax in axs[4:]:
        ax.set_xlabel("time [s]")

    fig.suptitle(
        "Ground reaction force per leg\n(moving average + Savitzky–Golay)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved contact force plot to {out_path}")

from scipy.signal import savgol_filter  # already imported in your script

def smooth_grf_logs(grf_logs, window_ma, window_sg, poly_sg):
    """
    Apply smoothing: moving average → SavGol to each leg in the dict.
    Returns a new dict with the smoothing applied.
    """
    smoothed = {}
    for leg, arr_list in grf_logs.items():
        forces = np.asarray(arr_list, dtype=float)
        if forces.size == 0:
            smoothed[leg] = forces
            continue

        # 1) Moving-average smoothing
        n = forces.size
        win_ma = min(window_ma, n)
        if win_ma > 1:
            kernel = np.ones(win_ma, dtype=float) / win_ma
            forces_ma = np.convolve(forces, kernel, mode="same")
        else:
            forces_ma = forces

        # 2) SavGol smoothing on top
        if forces_ma.size >= 5:
            win_sg = min(window_sg, forces_ma.size)
            if win_sg % 2 == 0:
                win_sg -= 1  # make odd
            if win_sg > poly_sg:
                smoothed_arr = savgol_filter(forces_ma, win_sg, poly_sg)
            else:
                smoothed_arr = forces_ma
        else:
            smoothed_arr = forces_ma

        smoothed[leg] = smoothed_arr

    return smoothed


def save_smoothed_grf_to_pickle(time_log, grf_logs,
                               out_path="smoothed_grf.pkl",
                               window_ma=100, window_sg=301, poly_sg=3):
    """
    Smooth GRF logs and save with time_log to a pickle for inspection later.
    """
    # Smooth signals
    smoothed = smooth_grf_logs(grf_logs, window_ma, window_sg, poly_sg)

    output = {
        "time": np.asarray(time_log, dtype=float),
        "grf": smoothed,
    }

    with open(out_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved smoothed GRF logs to {out_path}")


# -------------------------------------------------------
# Camera control
# -------------------------------------------------------
THORAX_BODY_NAME = "Thorax"
CAMDISTANCE = 6

def update_camera_to_face_thorax_flat(data, cam, thorax_id, distance=0.03):
    """
    Lock camera to rotate in AZIMUTH with Thorax, but keep elevation fixed.
    """
    pos = data.xpos[thorax_id].copy()
    R = data.xmat[thorax_id].reshape(3, 3)

    # forward axis = +x column of body rotation matrix
    forward = R[:, 0]
    # compute yaw direction on the horizontal plane only
    v2 = -forward[:2]
    v2 /= (np.linalg.norm(v2) + 1e-8)
    yaw = np.degrees(np.arctan2(v2[1], v2[0]) -30)

    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = pos
    cam.distance = distance
    cam.elevation = 0.0
    cam.azimuth = yaw

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)

    joint_info = build_joint_index(model)

    thorax_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, THORAX_BODY_NAME)
    if thorax_id < 0:
        raise ValueError(f"Body '{THORAX_BODY_NAME}' not found in model")

    lf_muscle_ids = np.array(
        [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in lf_muscle_names],
        dtype=int,
    )
    assert len(lf_muscle_ids) == 15, (
        f"Expected 15 LF muscle actuators, "
        f"got {len(lf_muscle_ids)} from model."
    )

    lf_tendon_ids = build_lf_tendon_ids(model, lf_tendon_names)

    floor_geom_id, leg_geom_ids, geomid_to_leg = build_leg_tarsus_geom_ids(model)

    legs = ["LF", "RF", "LM", "RM", "LH", "RH"]
    grf_logs = {leg: [] for leg in legs}
    time_log = []
    step_count = 0

    with viewer.launch_passive(model, data) as v:

        cam = v.cam  # this is an mj.MjvCamera

        # Initial tracking camera on Thorax
        cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = thorax_id
        cam.distance = CAMDISTANCE
        cam.elevation = 60.0
        cam.azimuth = 180.0

        for cycle in range(NUM_CYCLES):
            print(f"Starting cycle {cycle + 1}/{NUM_CYCLES}")
            for offset in range(segment_len):
                if not v.is_running():
                    save_contact_force_plot(time_log, grf_logs)
                    return

                frame_idx = START_FRAME + offset

                joint_targets = compute_joint_targets_from_file_nonlf(frame_idx)
                joint_targets = clip_targets_to_ranges(joint_targets, joint_info)

                data.qfrc_applied[:] = 0.0
                for jname, q_des in joint_targets.items():
                    info = joint_info[jname]
                    q_idx = info["qpos_idx"]
                    v_idx = info["qvel_idx"]

                    q = data.qpos[q_idx]
                    dq = data.qvel[v_idx]

                    if jname.startswith("joint_RFTibia") or jname.startswith("joint_RFTrochanter") or jname == "joint_RFTibia_pitch":
                        kp_used = 600.0
                        kd_used = 2.5
                    else:
                        kp_used = KP
                        kd_used = KD

                    tau = kp_used * (q_des - q) - kd_used * dq
                    data.qfrc_applied[v_idx] += tau

                act_idx = offset // FRAMES_PER_ACT
                act_idx = min(act_idx, T_muscle - 1)
                ctrl_values = np.clip(muscle_raw[act_idx], 0.0, 1.0)
                data.ctrl[lf_muscle_ids] = ctrl_values
                update_tendon_colors(model, lf_tendon_ids, ctrl_values)

                mj.mj_step(model, data)

                # Contact forces
                leg_forces = {leg: 0.0 for leg in legs}
                cf = np.zeros(6, dtype=float)
                for k in range(data.ncon):
                    c = data.contact[k]
                    g1 = c.geom1
                    g2 = c.geom2

                    if (g1 == floor_geom_id and g2 in geomid_to_leg) or \
                       (g2 == floor_geom_id and g1 in geomid_to_leg):

                        mj.mj_contactForce(model, data, k, cf)
                        f_mag = np.linalg.norm(cf[:3])

                        if g1 == floor_geom_id:
                            leg = geomid_to_leg[g2]
                        else:
                            leg = geomid_to_leg[g1]

                        leg_forces[leg] += f_mag

                for leg in legs:
                    grf_logs[leg].append(leg_forces[leg])

                time_log.append(step_count * model.opt.timestep)
                step_count += 1

                update_camera_to_face_thorax_flat(data, v.cam, thorax_id, distance=CAMDISTANCE)

                v.sync()

        print("Finished all cycles.")

    save_contact_force_plot(time_log, grf_logs)
    save_smoothed_grf_to_pickle(
        time_log, grf_logs,
        out_path="contact_forces_sim.pkl",
        window_ma=SMOOTH_WINDOW_STEPS,
        window_sg=SAVGOL_WINDOW,
        poly_sg=SAVGOL_POLY
    )

if __name__ == "__main__":
    main()
