from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from .Basedataset import BaseDataset, ChannelConfig
from os.path import basename, dirname, join, exists
from pathlib import Path

from sleepwalker.datasets.normalizer.EEGFilterNormalizer import EEGFilterNormalizer
from sleepwalker.datasets.normalizer.PulseFilterNormalizer import PulseFilterNormalizer
from sleepwalker.datasets.normalizer.RespirationFilterNormalizer import RespirationFilterNormalizer
from sleepwalker.datasets.normalizer.SaturationFilterNormalizer import SaturationFilterNormalizer
from sleepwalker.datasets.normalizer.SignalFilterNormalizer import SignalFilterNormalizer


HSP_CHANNEL_GROUPS = {
    "eeg": [
        "C3-M2",
        "C4-M1",
        "F3-M2",
        "F4-M1",
        "O1-M2",
        "O2-M1",
        "CZ-M2",
        "CZ-M1",
        "C3-M1",
        "C4-M2",
        "F3-M1",
        "F4-M2",
        "O1-M1",
        "O2-M2",
    ],
    "eog": ["E1-M2", "E2-M1", "E2-M2", "E1", "E2", "Eye Up", "Eye Down"],
    "chin_emg": [
        "CHIN1-CHIN2",
        "Chin1-Chin2",
        "CHIN1-CHIN3",
        "Chin1-Chin3",
        "CHIN2-CHIN3",
        "Chin2-Chin3",
        "Chin3-Chin2",
        "Chin3",
        "Chin2",
        "CHIN1",
        "CHIN2",
        "CHIN3",
        "CHIN",
        "CHINz",
        "EMG",
    ],
    "leg_emg": [
        "LAT",
        "RAT",
        "LLEG+",
        "LLEG-",
        "RLEG+",
        "RLEG-",
        "L LEG",
        "R LEG",
        "left leg",
        "right leg",
        # "Arm1",
        # "Arm2",
    ],
    "emg": [],
    "abdomen": [
        "ABD",
        "ABDOMEN",
        "Abdomen",
    ],
    "chest": [
        "CHEST",
        "THORAX",
        "Chest",
    ],
    "airflow": [
        "PTAF",
        "AIRFLOW",
        "AirFlow",
        "Airflow2",
        "IC",
        "THERMISTOR",
        "Thermistor",
        "Flow",
        "Flow_DR",
        "CFLOW",
        "CFlow",
        "C-Flow",
        "XFlow",
    ],
    "spo2": [
        "SaO2",
        "SpO2",
        "SPO2",
    ],
    "respiratory": [
        "ABD",
        "ABDOMEN",
        "Abdomen",
        "CHEST",
        "THORAX",
        "Chest",
        "PTAF",
        "AIRFLOW",
        "AirFlow",
        "Airflow2",
        "IC",
        "THERMISTOR",
        "Thermistor",
        "Flow",
        "Flow_DR",
        "CFLOW",
        "CFlow",
        "C-Flow",
        "XFlow",
        "SaO2",
        "SpO2",
        "SPO2",
    ],
    "pulse": [ 
        "EKG",
        "ECG",
        "ECG-LA",
        "ECG-RA",
        "ECG-LL",
        "ECG-V1",
        "ECG-V2",
        "Pleth",
        "PPG",
        "HR",
        "PR",
        "PulseQuality",
        # "pulse_rate_event",
    ],
}

HSP_CHANNEL_GROUPS["emg"] = HSP_CHANNEL_GROUPS["chin_emg"] + HSP_CHANNEL_GROUPS["leg_emg"]


def get_hsp_annotation_path(edf_path: str | Path) -> str | None:
    """Return the same-record HSP annotation sidecar for an EDF, if present."""
    edf_path = Path(edf_path)
    bn = edf_path.name
    candidates = [
        edf_path.with_name(bn.replace("eeg", "annotations").replace(".edf", ".csv")),
        edf_path.with_name(bn.replace("-psg_eeg.edf", "_Xltek.csv")),
    ]

    parts = edf_path.parts
    for idx, part in enumerate(parts):
        if part.startswith("sub-") and (idx == 0 or parts[idx - 1] != "HSP"):
            mirror_path = Path(*parts[:idx]) / "HSP" / Path(*parts[idx:])
            mirror_bn = mirror_path.name
            candidates.extend(
                [
                    mirror_path.with_name(mirror_bn.replace("eeg", "annotations").replace(".edf", ".csv")),
                    mirror_path.with_name(mirror_bn.replace("-psg_eeg.edf", "_Xltek.csv")),
                ]
            )
            break

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def hsp_record_key(edf_path: str | Path) -> str:
    """Return a path-independent key for one HSP subject/session recording."""
    path = Path(edf_path)
    subject = next((part for part in path.parts if part.startswith("sub-")), None)
    session = next((part for part in path.parts if part.startswith("ses-")), None)
    if subject is None or session is None:
        raise ValueError(f"Expected HSP path with sub-* and ses-* components, got: {path}")
    return f"{subject}/{session}/{path.name}"


def _hsp_path_preference(edf_path: str) -> tuple[int, str]:
    """Prefer the direct subject tree over the optional ``HSP/`` mirror."""
    parts = Path(edf_path).parts
    subject_idx = next((idx for idx, part in enumerate(parts) if part.startswith("sub-")), None)
    mirrored = subject_idx is not None and subject_idx > 0 and parts[subject_idx - 1] == "HSP"
    return (int(mirrored), edf_path)


def get_annotated_hsp_edf_files(root: str | Path, recursive: bool = True) -> list[str]:
    """List unique HSP EDF records that have an annotation sidecar."""
    from sleepwalker.datasets.utils import get_edf_files_in_repo

    records: dict[str, str] = {}
    for edf_path in sorted(
        get_edf_files_in_repo(str(root), recursive=recursive),
        key=_hsp_path_preference,
    ):
        if get_hsp_annotation_path(edf_path) is None:
            continue
        records.setdefault(hsp_record_key(edf_path), edf_path)
    return sorted(records.values())


def map_hsp_sane_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map heterogeneous HSP annotation labels to stable task labels."""
    df = df.copy()
    exact_mapping = {
        'stage - n1': 'n1',
        'sleep_stage_n1': 'n1',
        'sleep_stage_1': 'n1',
        'stage - n2': 'n2',
        'sleep_stage_n2': 'n2',
        'sleep_stage_2': 'n2',
        'stage - n3': 'n3',
        'sleep_stage_n3': 'n3',
        'sleep_stage_3': 'n3',
        'stage - r': 'rem',
        'sleep_stage_r': 'rem',
        'sleep_stage_rem': 'rem',
        'rem': 'rem',
        'stage - w': 'wake',
        'sleep_stage_w': 'wake',
        'apnea / desats': 'apnea',
        'oxygen_desaturation': 'desaturation'
    }
    df['Label'] = df['Label'].replace(exact_mapping)

    substring_matches = {
        'desaturation': 'desaturation',
        'obstructive apnea': 'obstructive-apnea',
        'obstructive_apnea': 'obstructive-apnea',
        'obstructiveapnea': 'obstructive-apnea',
        'mixed apnea': 'mixed-apnea',
        'mixed_apnea': 'mixed-apnea',
        'mixedapnea': 'mixed-apnea',
        'central apnea': 'central-apnea',
        'central_apnea': 'central-apnea',
        'centralapnea': 'central-apnea',
        'hypopnea': 'hypopnea',
        'rera': 'rera',
        'arousal':'arousal'
    }
    orig = df['Label']
    result = orig.copy()
    for k, v in substring_matches.items():
        mask = orig.str.contains(k, case=False, na=False)
        result = result.mask(mask, v)
    df['Label'] = result
    return df


def get_hsp_annotation_label_counts(annotation_path: str | Path) -> dict[str, int]:
    """Return sane-label counts for an HSP annotation sidecar."""
    df = pd.read_csv(annotation_path)
    df = df.rename(columns={"event": "Label"})
    if "Label" not in df.columns:
        raise ValueError(f"Annotation file has no Label/event column: {annotation_path}")
    df["Label"] = df["Label"].astype(str).str.lower()
    df = map_hsp_sane_labels(df)
    return df["Label"].value_counts().sort_index().astype(int).to_dict()


def hsp_normalizer(channel_name: str, sample_frequency: float):
    """Return a convenience normalizer for an HSP channel.

    The defaults below mirror the pragmatic Ruhrlandklinik helper: EEG/EOG use
    the EEG filter stack, respiratory channels use the respiration filter,
    oxygen saturation uses the saturation filter, and EMG/pulse-like channels
    get lightweight band-pass defaults. Unknown channels fall back to ``None``.
    """
    if channel_name in HSP_CHANNEL_GROUPS["eeg"] or channel_name in HSP_CHANNEL_GROUPS["eog"]:
        return EEGFilterNormalizer(fs=sample_frequency)
    if channel_name in HSP_CHANNEL_GROUPS["chin_emg"] or channel_name in HSP_CHANNEL_GROUPS["leg_emg"]:
        return SignalFilterNormalizer(fs=sample_frequency, lowcut=10.0, highcut=45.0, notch_freq=60.0)
    if channel_name in {"SaO2", "SpO2", "SPO2"}:
        return SaturationFilterNormalizer(fs=sample_frequency)
    if channel_name in {"Pleth", "PPG"}:
        return PulseFilterNormalizer(fs=sample_frequency)
    if channel_name in HSP_CHANNEL_GROUPS["respiratory"]:
        return RespirationFilterNormalizer(fs=sample_frequency)
    if channel_name in {"EKG"}:
        return SignalFilterNormalizer(fs=sample_frequency, lowcut=0.5, highcut=8.0)
    return None


def hsp_group_name(subgroup: str, grouped: bool) -> str | None:
    """Resolve the logical group label used for grouped channel sampling."""
    if not grouped:
        return None
    if subgroup == "eeg":
        return "EEG"
    if subgroup == "eog":
        return "EOG"
    if subgroup == "chin_emg":
        return "Chin EMG"
    if subgroup == "leg_emg":
        return "Leg EMG"
    if subgroup == "abdomen":
        return "Abdomen"
    if subgroup == "chest":
        return "Chest"
    if subgroup == "airflow":
        return "Airflow"
    if subgroup == "spo2":
        return "SpO2"
    if subgroup == "respiratory":
        return "Respiratory"
    if subgroup == "pulse":
        return "Pulse"
    return None

def resolve_normalizer(
    channel_name: str,
    group_name: str | None,
    normalize: bool,
    override_normalize: dict[str, object | None] | None,
    sample_frequency: float,
):
    """Resolve the effective normalizer for one HSP channel."""
    if override_normalize is not None:
        if channel_name in override_normalize:
            return override_normalize[channel_name]
        if group_name is not None and group_name in override_normalize:
            return override_normalize[group_name]
    if not normalize:
        return None
    return hsp_normalizer(channel_name, sample_frequency)


def get_channels(
    group_names: list[str],
    grouped: bool,
    normalize: bool,
    sample_frequency: float,
    override_normalize: dict[str, object | None] | None = None,
) -> list[ChannelConfig]:
    """Build HSP channel configurations from groups and/or channel names.

    This mirrors the Ruhrlandklinik helper so training scripts can use the same
    calling pattern across datasets. The channel groups are derived from the HSP
    channel inventory documented in the class docstring below and intentionally
    prefer the most common referenced PSG montage names.
    """
    requested = []
    seen_names = set()
    valid_groups = set(HSP_CHANNEL_GROUPS.keys())

    for raw_group_name in group_names:
        group_name = raw_group_name.lower().strip()
        if group_name in valid_groups:
            if group_name == "emg":
                subgroups = ["chin_emg", "leg_emg"]
            else:
                subgroups = [group_name]

            for subgroup in subgroups:
                logical_group = hsp_group_name(subgroup, grouped)
                for channel_name in HSP_CHANNEL_GROUPS[subgroup]:
                    if channel_name in seen_names:
                        continue
                    seen_names.add(channel_name)
                    requested.append(
                        ChannelConfig(
                            name=channel_name,
                            normalizer=resolve_normalizer(
                                channel_name=channel_name,
                                group_name=subgroup,
                                normalize=normalize,
                                override_normalize=override_normalize,
                                sample_frequency=sample_frequency,
                            ),
                            group=logical_group,
                            unit="%" if subgroup == "spo2" else "uV",
                        )
                    )
            continue

        channel_name = raw_group_name
        if channel_name in seen_names:
            continue
        seen_names.add(channel_name)
        requested.append(
            ChannelConfig(
                name=channel_name,
                normalizer=resolve_normalizer(
                    channel_name=channel_name,
                    group_name=None,
                    normalize=normalize,
                    override_normalize=override_normalize,
                    sample_frequency=sample_frequency,
                ),
                group=None,
                unit="%" if channel_name in HSP_CHANNEL_GROUPS["spo2"] else "uV",
            )
        )

    return requested

class HSP(BaseDataset):
    '''
    Summary of core statistics for this dataset.

        Durations:
            min     : 0 days 00:01:30
            max     : 0 days 12:41:44
            mean    : 0 days 07:28:15.480000
            median  : 0 days 07:41:11
            q25     : 0 days 07:06:39
            q75     : 0 days 08:08:09

        Channels (244 total):
              1. ABD                       97.9%
              2. CHEST                     97.6%
              3. E1-M2                     97.4%
              4. C3-M2                     96.5%
              5. O1-M2                     96.5%
              6. F3-M2                     96.0%
              7. C4-M1                     95.9%
              8. O2-M1                     95.7%
              9. F4-M1                     95.6%
             10. PTAF                      93.1%
             11. IC                        83.9%
             12. EKG                       83.2%
             13. LAT                       82.5%
             14. RAT                       82.3%
             15. SNORE                     81.8%
             16. SaO2                      80.3%
             17. HR                        80.1%
             18. AIRFLOW                   79.0%
             19. CHIN1-CHIN2               59.5%
             20. CFLOW                     57.6%
             21. E2-M1                     54.3%
             22. Leak                      51.1%
             23. E2-M2                     43.3%
             24. CPRES                     34.6%
             25. Chin1-Chin2               34.1%
             26. EtCO2                     25.5%
             27. DC8                       20.0%
             28. P4                        19.7%
             29. M2                        19.7%
             30. P3                        19.7%
             31. M1                        19.7%
             32. PR                        19.7%
             33. DC11                      19.7%
             34. DC9                       19.7%
             35. F7                        19.7%
             36. F8                        19.7%
             37. Pleth                     19.7%
             38. DC10                      19.7%
             39. Fp1                       19.5%
             40. Fp2                       19.5%
             41. DC7                       19.2%
             42. AirFlow                   18.6%
             43. CO2 Wave                  18.6%
             44. LEAK                      18.5%
             45. SpO2                      18.2%
             46. DC14                      18.1%
             47. Snore                     18.0%
             48. CPAP                      17.3%
             49. DC1                       17.2%
             50. T7                        17.0%
             51. Fz                        17.0%
             52. Cz                        17.0%
             53. XFlow                     17.0%
             54. DC16                      17.0%
             55. PTT                       17.0%
             56. DC15                      17.0%
             57. PPG                       17.0%
             58. Position                  17.0%
             59. Elevation                 17.0%
             60. Activity                  17.0%
             61. TRIG                      17.0%
             62. Pz                        17.0%
             63. P7                        17.0%
             64. Oz                        17.0%
             65. Fpz                       17.0%
             66. P8                        17.0%
             67. T8                        17.0%
             68. X32                       17.0%
             69. Phase                     17.0%
             70. Pressure                  17.0%
             71. DIF7-                     17.0%
             72. DIF1+                     17.0%
             73. DIF2-                     17.0%
             74. DIF3+                     17.0%
             75. DIF2+                     17.0%
             76. DIF1-                     17.0%
             77. DIF5-                     17.0%
             78. DIF6+                     17.0%
             79. DIF4-                     17.0%
             80. DIF5+                     17.0%
             81. DIF6-                     17.0%
             82. DIF7+                     17.0%
             83. DIF4+                     17.0%
             84. DIF3-                     17.0%
             85. PulseQuality              17.0%
             86. DC12                      17.0%
             87. CFlow                     17.0%
             88. DIF8-                     17.0%
             89. DIF8+                     17.0%
             90. RR                        17.0%
             91. XVolume                   17.0%
             92. RMI                       17.0%
             93. XSum                      17.0%
             94. ECG-V2                    17.0%
             95. ECG-V1                    17.0%
             96. ECG-LL                    17.0%
             97. DIF9+                     16.7%
             98. IPAP                      16.7%
             99. DIF10-                    16.7%
            100. DIF10+                    16.7%
            101. DIF9-                     16.7%
            102. EPAP                      16.4%
            103. Eye Down                  16.1%
            104. Eye Up                    16.1%
            105. Snore_DR                  16.0%
            106. CHIN3                     15.9%
            107. Arm2                      15.9%
            108. Arm1                      15.9%
            109. CHIN2                     15.3%
            110. ECG-LA                    15.3%
            111. ECG-RA                    15.3%
            112. RLEG+                     15.3%
            113. LLEG-                     15.3%
            114. LLEG+                     15.3%
            115. RLEG-                     15.3%
            116. IC2                       14.2%
            117. IC1                       14.2%
            118. Airflow2                  14.0%
            119. Tidal                     13.9%
            120. ASVFL                     3.7%
            121. DC6                       3.5%
            122. CZ-M2                     2.9%
            123. Chin3                     2.7%
            124. Chin2                     2.7%
            125. T5                        2.7%
            126. T3                        2.7%
            127. 30                        2.7%
            128. T6                        2.7%
            129. T4                        2.7%
            130. C4                        1.9%
            131. O1                        1.9%
            132. C3                        1.9%
            133. E2                        1.9%
            134. F4                        1.9%
            135. F3                        1.9%
            136. E1                        1.9%
            137. O2                        1.9%
            138. Chin1-Chin3               1.8%
            139. .                         1.7%
            140. EMG                       1.7%
            141. L LEG                     1.7%
            142. CHIN                      1.7%
            143. R LEG                     1.7%
            144. ,,                        1.7%
            145. ,                         1.7%
            146. ABDOMEN                   1.7%
            147. THORAX                    1.7%
            148. ..                        1.7%
            149. THERMISTOR                1.7%
            150. CLEAK                     1.6%
            151. CHIN1-CHIN3               1.5%
            152. ECG                       1.5%
            153. SPO2                      1.5%
            154. O2-E1                     1.3%
            155. F4-M2                     1.2%
            156. C4-M2                     1.2%
            157. O2-M2                     1.2%
            158. DC13                      1.2%
            159. X29                       1.1%
            160. X25                       1.1%
            161. X30                       1.1%
            162. X31                       1.1%
            163. Flow                      1.1%
            164. X28                       1.1%
            165. X26                       1.1%
            166. CHINz                     1.1%
            167. DC2                       1.1%
            168. Flow_DR                   1.1%
            169. SNORE.DR                  1.0%
            170. leak                      1.0%
            171. C-Leak                    0.9%
            172. C-Flow                    0.9%
            173. C-Pres                    0.9%
            174. CZ-M1                     0.9%
            175. Cap Wave                  0.9%
            176. F4-AVG                    0.8%
            177. C3-M1                     0.8%
            178. T3-M1                     0.8%
            179. F3-M1                     0.8%
            180. O1-M1                     0.8%
            181. O2-AVG                    0.8%
            182. O1-AVG                    0.8%
            183. C4-AVG                    0.8%
            184. 31-32                     0.8%
            185. C3-AVG                    0.8%
            186. T4-M2                     0.8%
            187. F3-AVG                    0.8%
            188. Fp1-M2                    0.6%
            189. F8-M2                     0.6%
            190. T5-M1                     0.6%
            191. F7-M1                     0.6%
            192. T6-M2                     0.6%
            193. Fp2-M1                    0.5%
            194. E2-AVG                    0.5%
            195. E1-AVG                    0.5%
            196. OLD CPAP                  0.5%
            197. LEA                       0.4%
            198. PRESS                     0.4%
            199. CHIN2-CHIN3               0.4%
            200. T6-M1                     0.4%
            201. F7-M2                     0.4%
            202. Chin3-Chin2               0.3%
            203. R ARM                     0.3%
            204. .,                        0.3%
            205. CAPNO                     0.3%
            206. ,.                        0.3%
            207. DC5                       0.3%
            208. L ARM                     0.3%
            209. DC4                       0.2%
            210. RexDIG1--2                0.2%
            211. LexDIG1-2                 0.2%
            212. CHIN1                     0.2%
            213. O1-E1                     0.2%
            214. Abdomen                   0.2%
            215. Chest                     0.2%
            216. DC3                       0.2%
            217. ETC02                     0.2%
            218. P4-M1                     0.2%
            219. P3-M2                     0.2%
            220. CHIN3-CHIN2               0.2%
            221. T4-M1                     0.2%
            222. T3-M2                     0.2%
            223. T5-M2                     0.2%
            224. F8-M1                     0.2%
            225. CZ-C4                     0.2%
            226. C3-CZ                     0.2%
            227. PZ-M2                     0.2%
            228. FZ-M2                     0.2%
            229. Chin2-Chin3               0.2%
            230. TrgCy                     0.2%
            231. FZ-E1                     0.1%
            232. O2-T5                     0.1%
            233. Thermistor                0.1%
            234. F4-T5                     0.1%
            235. C4-T5                     0.1%
            236. CHIN3-CHIN1               0.1%
            237. Leak-1                    0.1%
            238. Leak-0                    0.1%
            239. EtC02                     0.1%
            240. 31-31                     0.1%
            241. CZ-AVG                    0.1%
            242. F4-CZ                     0.1%
            243. C4-CZ                     0.1%
            244. O2-CZ                     0.1%

        Classes (n>1000):
            - sleep_stage_n2 - 165344825
            - sleep_stage_w - 69064497
            - sleep_stage_n3 - 47194342
            - sleep_stage_n1 - 45578656
            - sleep_stage_r - 27911864
            - plm - periodic - 21784824
            - sleep_stage_rem - 20241121
            - plm - periodic - 1 - 17688951
            - redacted - 17661939
            - * arousal - respiratory event - 14315079
            - limb_movement - 13826377
            - sleep_stage_? - 10873267
            - respevent - rera - 6 - 5972527
            - respevent - hypopnea - 4 - 5940806
            - * arousal - spontaneous - 4462517
            - oxygen_desaturation - 4380591
            - respevent - obstructiveapnea - 1 - 3442510
            - plm - isolated - 3437085
            - * arousal - plm - 3412040
            - respevent - centralapnea - 2 - 2637210
            - plm - isolated - 1 - 2320224
            - hypopnea - 2295963
            - obstructive_apnea - 1247926
            - central_apnea - 1148655
            - respiratory event - rera - desat 95.0 % - 1072245
            - respiratory event - rera - desat 94.0 % - 1070802
            - respiratory event - hypopnea - desat 91.0 % - 1002307
            - respiratory event - hypopnea - desat 92.0 % - 981565
            - rera - 897583
            - respiratory event - rera - desat 93.0 % - 852135
            - respiratory event - hypopnea - desat 93.0 % - 820929
            - respiratory event - hypopnea - desat 90.0 % - 802673
            - arousal - respevent - 1 - 774885
            - respiratory event - rera - desat 96.0 % - 744988
            - position - supine - 734541
            - impedance check - 721271
            - cough - 669106
            - look right - 650310
            - look left - 645306
            - spo2_threshold_event - 633720
            - arousal - respiratory event - 631105
            - look up - 621246
            - position - supine - 1 - 613771
            - look down - 613077
            - respiratory event - rera - desat 92.0 % - 602247
            - respiratory event - hypopnea - desat 89.0 % - 593596
            - snore - 585506
            - * desaturation - min 92.0 % - drop  4.2 % - 573714
            - respiratory event - hypopnea - desat 94.0 % - 539029
            - respiratory event - central apnea - desat 92.0 % - 519758
            - * desaturation - min 93.0 % - drop  4.1 % - 517795
            - respiratory event - central apnea - desat 93.0 % - 492989
            - * desaturation - min 94.0 % - drop  4.1 % - 491478
            - * desaturation - min 91.0 % - drop  4.2 % - 461808
            - respiratory event - central apnea - desat 91.0 % - 432881
            - respiratory event - rera - desat 97.0 % - 426934
            - pulse_rate_event - 419968
            - respiratory event - hypopnea - desat 88.0 % - 403588
            - * desaturation - min 91.0 % - drop  5.2 % - 403132
            - * desaturation - min 93.0 % - drop  5.1 % - 399948
            - new montage - 1) standard montage - 383505
            - respiratory event - central apnea - desat 90.0 % - 380907
            - respiratory event - central apnea - desat 94.0 % - 379759
            - blink - 376798
            - snore - isolated - 376038
            - * desaturation - min 92.0 % - drop  5.2 % - 369597
            - right leg - 359927
            - lights on - 359844
            - lights out - 354396
            - position - right - 353860
            - body_position:_supine - 353676
            - position - left - 353319
            - left leg - 353242
            - respiratory event - rera - desat 91.0 % - 347145
            - respiratory event - rera - desat 98.0 % - 341949
            - sleep_stage_2 - 337193
            - respiratory event - central apnea - desat 95.0 % - 331999
            - respevent - mixedapnea - 3 - 329891
            - respiratory event - central apnea - desat 89.0 % - 325029
            - respiratory event - obstructive apnea - desat 92.0 % - 316134
            - * arousal - snore - 315669
            - hold breath - 309716
            - respiratory event - obstructive apnea - desat 93.0 % - 300470
            - position - right - 1 - 294603
            - respiratory event - obstructive apnea - desat 91.0 % - 292767
            - position - left - 1 - 289595
            - * snore - periodic - 287620
            - respiratory event - obstructive apnea - desat 90.0 % - 285255
            - * desaturation - min 92.0 % - drop  6.1 % - 285087
            - * desaturation - min 90.0 % - drop  6.3 % - 284414
            - mouth breath - 281744
            - * desaturation - min 90.0 % - drop  5.3 % - 281169
            - * arousal - dur:  3.0 sec. - respiratory event - 269332
            - respiratory event - hypopnea - desat 87.0 % - 268605
            - * desaturation - min 90.0 % - drop  4.3 % - 264189
            - * desaturation - min 91.0 % - drop  6.2 % - 247858
            - respiratory event - obstructive apnea - desat 94.0 % - 245729
            - movement - 245408
            - respiratory event - obstructive apnea - desat 89.0 % - 238345
            - * desaturation - min 91.0 % - drop  7.1 % - 229586
            - snore - isolated - 1 - 227575
            - respiratory event - central apnea - desat 88.0 % - 224622
            - * desaturation - min sao2 92.0 % - 218410
            - respiratory event - central apnea - desat 96.0 % - 213620
            - respiratory event - hypopnea - desat 86.0 % - 202469
            - * desaturation - min 89.0 % - drop  7.3 % - 198686
            - pulse_rate_event:_pulse_search - 194768
            - respiratory event - obstructive apnea - desat 95.0 % - 194217
            - treatment - none - 193639
            - * desaturation - min 89.0 % - drop  6.3 % - 188121
            - * desaturation - min sao2 90.0 % - 187320
            - body_position:_right - 185729
            - * desaturation - min 90.0 % - drop  7.2 % - 185615
            - rem - 185366
            - start recording - 184229
            - respiratory event - obstructive apnea - desat 88.0 % - 183965
            - spo2_event:_pulse_search - 182535
            - * desaturation - min sao2 93.0 % - 181953
            - * desaturation - min sao2 89.0 % - 180025
            - new montage - 1a) standard montage with cpap - 179579
            - look_left - 179409
            - look_down - 177869
            - * desaturation - min sao2 91.0 % - 177643
            - body_position:_left - 177592
            - look_up - 177510
            - look_right - 177095
            - respiratory event - rera - desat 90.0 % - 172717
            - * desaturation - min 90.0 % - drop  8.2 % - 172203
            - pulse_rate_threshold_event - 171498
            - respiratory event - central apnea - desat 87.0 % - 171470
            - impedance_at_10_kohm - 166815
            - * desaturation - min 89.0 % - drop  5.3 % - 158595
            - treatment - none - 1 - 155948
            - respiratory event - obstructive apnea - desat 87.0 % - 155529
            - * desaturation - min sao2 88.0 % - 155085
            - stage - n2 - 154800
            - respiratory event - hypopnea - desat 85.0 % - 153468
            - respiratory event - hypopnea - desat 95.0 % - 153093
            - stage - n1 - 152039
            - blink_5x - 146910
            - respiratory event - central apnea - desat 97.0 % - 146523
            - mixed_apnea - 145805
            - hold_breath - 145038
            - nasal_breath - 144954
            - oral_breath - 143076
            - bilevel - applied - ipap   0.0 cmh2o - epap   0.0 cmh2o - 141684
            - nasal breath - 137740
            - * desaturation - min 89.0 % - drop  4.3 % - 131731
            - * desaturation - min 89.0 % - drop  8.2 % - 126928
            - * desaturation - min 88.0 % - drop  8.3 % - 125567
            - * desaturation - min 89.0 % - drop  9.2 % - 123072
            - respiratory event - hypopnea - desat 84.0 % - 122162
            - ekgevent - sinustach - 1 - 120672
            - desaturation - 1 - 117966
            - respiratory event - obstructive apnea - desat 86.0 % - 117366
            - * desaturation - min sao2 87.0 % - 114726
            - sleep_stage_1 - 113829
            - * desaturation - min sao2 94.0 % - 112671
            - * desaturation - min 88.0 % - drop  7.4 % - 109194
            - * desaturation - min 88.0 % - drop  9.3 % - 107809
            - respiratory event - rera - desat  0.0 % - 105057
            - respiratory event - obstructive apnea - desat 96.0 % - 104040
            - * desaturation - min 95.0 % - drop  4.0 % - 100577
            - respiratory event - obstructive apnea - desat 85.0 % - 99514
            - impedance_at_25_kohm - 98525
            - respiratory event - central apnea - desat 98.0 % - 98427
            - respiratory event - hypopnea - desat 83.0 % - 97356
            - respiratory event - central apnea - desat 86.0 % - 97096
            - bilevel - applied - ipap   6.0 cmh2o - epap   6.0 cmh2o - 94504
            - * desaturation - min 87.0 % - drop  9.4 % - 93434
            - bilevel - applied - ipap   7.0 cmh2o - epap   7.0 cmh2o - 92356
            - * arousal - dur:  3.0 sec. - plm - 90728
            - treatment - start - 90503
            - bilevel - applied - ipap   5.0 cmh2o - epap   5.0 cmh2o - 90309
            - * desaturation - min sao2 86.0 % - 89142
            - leg movement - 88963
            - respiratory event - obstructive apnea - desat  0.0 % - 85989
            - * desaturation - min 88.0 % - drop  6.4 % - 85967
            - * arousal - dur:  3.0 sec. - spontaneous - 85107
            - lights_off - 84827
            - lights_on - 84674
            - respiratory event - obstructive apnea - desat 84.0 % - 84262
            - * desaturation - min 88.0 % - drop  5.4 % - 82874
            - bilevel - applied - ipap   4.0 cmh2o - epap   4.0 cmh2o - 82588
            - bilevel - applied - ipap   8.0 cmh2o - epap   8.0 cmh2o - 81637
            - video_recording_on - 80615
            - * desaturation - min sao2 85.0 % - 79303
            - respiratory event - rera - desat 89.0 % - 79108
            - new sensitivity 4000 µvp-p abd - 78776
            - stage - w - 78766
            - treatment - start - 1 - 78665
            - * desaturation - min 87.0 % - drop  8.4 % - 78283
            - * desaturation - min 88.0 % - drop 10.2 % - 77663
            - * desaturation - min sao2 83.0 % - 77455
            - waveform polarity changed to inverted: abd - 77239
            - respiratory event - central apnea - desat 85.0 % - 76737
            - new montage - 2) standard montage with cpap - 75531
            - respiratory event - hypopnea - desat 82.0 % - 74784
            - respiratory event - obstructive apnea - desat 83.0 % - 74766
            - * desaturation - min sao2 84.0 % - 73993
            - new sensitivity 5000 µvp-p abd - 73393
            - * desaturation - min 88.0 % - drop  4.3 % - 71501
            - respiratory event - obstructive apnea - desat 81.0 % - 70968
            - respiratory event - obstructive apnea - desat 82.0 % - 70894
            - new sensitivity 2000 µvp-p chest - 70195
            - * desaturation - min 87.0 % - drop 10.3 % - 70089
            - respiratory event - obstructive apnea - desat 80.0 % - 69657
            - new sensitivity 1500 µvp-p chest - 68744
            - new sensitivity 10000 µvp-p abd - 67211
            - started_analyzer_-_sleep_events - 66966
            - pt to bathroom - 66444
            - new sensitivity 3000 µvp-p chest - 66298
            - bilevel - applied - ipap   9.0 cmh2o - epap   9.0 cmh2o - 65659
            - waveform polarity changed to inverted: chest - 64354
            - new sensitivity 4000 µvp-p chest - 64206
            - rem supine - 62992
            - ekg events - sinus tachycardia - 60834
            - new sensitivity 3000 µvp-p abd - 58353
            - respiratory event - obstructive apnea - desat 79.0 % - 57884
            - plm - dur:  1.0 sec. - periodic - 57490
            - bruxism - isolated - 57348
            - stopped_analyzer_-_sleep_events - 57015
            - respiratory event - obstructive apnea - desat 76.0 % - 56922
            - new sensitivity 1000 µvp-p chest - 55324
            - respiratory event - central apnea - desat  0.0 % - 55221
            - * desaturation - min 94.0 % - drop  5.1 % - 55052
            - * desaturation - min 86.0 % - drop 10.4 % - 54752
            - * desaturation - min 87.0 % - drop  7.4 % - 54582
            - respiratory event - hypopnea - desat 81.0 % - 53511
            - waveform polarity changed to normal: abd - 52597
            - * desaturation - min sao2 82.0 % - 52521
            - * desaturation - min 87.0 % - drop 11.2 % - 51934
            - respiratory event - obstructive apnea - desat 77.0 % - 51659
            - nasal - 51253
            - stage - n3 - 50841
            - arousal - spontaneous - 1 - 50816
            - respiratory event - obstructive apnea - desat 75.0 % - 50442
            - * desaturation - min 86.0 % - drop  9.5 % - 50162
            - bilevel - applied - ipap  10.0 cmh2o - epap  10.0 cmh2o - 49185
            - montage:01_standard_montage_aasm_[01] - 48249
            - respiratory event - obstructive apnea - desat 78.0 % - 48061
            - respiratory event - obstructive apnea - desat 97.0 % - 46348
            - * desaturation - min sao2 81.0 % - 46310
            - waveform polarity changed to normal: chest - 46276
            - * desaturation - min 86.0 % - drop 11.3 % - 45329
            - respiratory event - hypopnea - desat  0.0 % - 45185
            - montage:channel_test_referential - 44117
            - * desaturation - min 87.0 % - drop  6.5 % - 44114
            - respiratory event - obstructive apnea - desat 98.0 % - 43959
            - new sensitivity 2000 µvp-p abd - 43701
            - respiratory event - central apnea - desat 84.0 % - 43594
            - respiratory event - hypopnea - desat 96.0 % - 43518
            - respiratory event - hypopnea - desat 80.0 % - 43126
            - respiratory event - central apnea - desat 83.0 % - 43126
            - * desaturation - min 93.0 % - drop  6.1 % - 41921
            - new sensitivity 750µv chest - 41765
            - cpap:6_cmh2o - 41404
            - bruxism - isolated - 1 - 40808
            - de-block_start - 40695
            - analyzer_montage_changed_-_ecg - 40284
            - ekg events - cardiac event - 40205
            - recording_analyzer_-_data_trends - 39788
            - cpap:7_cmh2o - 39055
            - * desaturation - min 85.0 % - drop 11.5 % - 38620
            - * desaturation - min 86.0 % - drop 12.2 % - 38593
            - cpap:4_cmh2o - 38481
            - arousal - 38415
            - * desaturation - min 85.0 % - drop 12.4 % - 37212
            - * desaturation - min 87.0 % - drop  5.4 % - 37013
            - respiratory event - obstructive apnea - desat 74.0 % - 36590
            - * desaturation - min 86.0 % - drop  8.5 % - 36239
            - cpap:5_cmh2o - 36162
            - new sensitivity 1000µv chest - 35884
            - * desaturation - min 85.0 % - drop 13.3 % - 35807
            - bilevel - applied - ipap  11.0 cmh2o - epap  11.0 cmh2o - 35530
            - analyzer_montage_changed_-_data_trends - 35156
            - respiratory event - hypopnea - desat 79.0 % - 34581
            - recording_analyzer_-_ecg - 34176
            - new sensitivity 500µv chest - 33936
            - cpap:8_cmh2o - 33924
            - arousal - plm - 33423
            - new sensitivity 10000 µvp-p chest - 33403
            - respiratory event - central apnea - desat 82.0 % - 33322
            - respiratory event - central apnea - desat 81.0 % - 33268
            - * desaturation - min 87.0 % - drop  4.4 % - 32962
            - new sensitivity 3000 µvp-p airflow - 32464
            - stage - r - 32293
            - plm - dur:  1.0 sec. - isolated - 31494
            - new sensitivity 4000 µvp-p airflow - 31199
            - arousal - other - 31152
            - respiratory event - rera - desat 88.0 % - 31073
            - * desaturation - min 85.0 % - drop 10.5 % - 30842
            - * desaturation - min 92.0 % - drop  7.1 % - 30797
            - respiratory event - hypopnea - desat 78.0 % - 30607
            - new sensitivity 5000 µvp-p chest - 29943
            - new sensitivity 1500 µvp-p abd - 29599
            - * desaturation - min 84.0 % - drop 12.5 % - 29569
            - * desaturation - min 86.0 % - drop  6.5 % - 29536
            - * desaturation - min 86.0 % - drop  7.5 % - 29111
            - * desaturation - min sao2 80.0 % - 28975
            - * desaturation - min 83.0 % - drop 14.4 % - 28880
            - * desaturation - min 84.0 % - drop 14.3 % - 28842
            - arousal - bruxism - 28820
            - new sensitivity 1500 µvp-p airflow - 28193
            - start_recording - 28067
            - rem_supine - 27464
            - cpap:9_cmh2o - 26852
            - * desaturation - min 82.0 % - drop 16.3 % - 26608
            - * desaturation - min 85.0 % - drop  9.6 % - 26543
            - bilevel - applied - ipap  12.0 cmh2o - epap  12.0 cmh2o - 26326
            - respiratory event - hypopnea - desat 77.0 % - 26192
            - cpap:10_cmh2o - 26165
            - * desaturation - min 84.0 % - drop 13.4 % - 25540
            - new sensitivity 750 µvp-p chest - 25350
            - new sensitivity 1000µv abd - 25298
            - recording resumed - 25265
            - pulse_rate_event:_low_quality - 25145
            - position - prone - 25102
            - new sensitivity 1000µv airflow - 25059
            - * desaturation - min 80.0 % - drop 16.7 % - 24909
            - * desaturation - min 81.0 % - drop 17.3 % - 24618
            - arousal - plm - 1 - 24282
            - new sensitivity 2000 µvp-p airflow - 24257
            - arousal - snore - 23985
            - * desaturation - min 83.0 % - drop 13.5 % - 23919
            - wake - 23681
            - respiratory event - rera - desat 99.0 % - 23442
            - * desaturation - min 81.0 % - drop 15.6 % - 23417
            - respiratory event - obstructive apnea - desat 73.0 % - 23306
            - * desaturation - min 84.0 % - drop 10.6 % - 23236
            - new sensitivity 1000 µvp-p airflow - 22891
            - respiratory event - central apnea - desat 79.0 % - 22522
            - ekg events - asystole - 21885
            - new sensitivity 500µv airflow - 21770
            - bruxism - 21708
            - new sensitivity 750µv airflow - 21679
            - * desaturation - min 83.0 % - drop 12.6 % - 21612
            - * desaturation - min 82.0 % - drop 14.6 % - 21568
            - lm - periodic - 21435
            - montage:01_standard_montage_aasm_[01]_ref - 21339
            - new sensitivity 300µv chest - 21330
            - bilevel - applied - ipap  13.0 cmh2o - epap  13.0 cmh2o - 21274
            - new sensitivity 1500µv chest - 21263
            - respiratory event - mixed apnea - desat 76.0 % - 20921
            - spo2_event:_sensor_off - 20666
            - respiratory event - central apnea - desat 99.0 % - 20487
            - * desaturation - min sao2 79.0 % - 20457
            - stop_recording - 20404
            - new sensitivity 300µv airflow - 20224
            - * desaturation - min 84.0 % - drop 11.6 % - 20213
            - waveform polarity changed to inverted: airflow - 20058
            - nose breath - 20058
            - * desaturation - min 83.0 % - drop 15.3 % - 19934
            - waveform polarity changed to inverted: ekg - 19910
            - * desaturation - min 91.0 % - drop  8.1 % - 19783
            - respiratory event - hypopnea - desat 76.0 % - 19761
            - new sensitivity 1000 µvp-p abd - 19695
            - position - disconnect - 1 - 19572
            - respiratory event - mixed apnea - desat 78.0 % - 19570
            - respiratory event - mixed apnea - desat 79.0 % - 19515
            - arousal - spontaneous - 19477
            - * desaturation - min 84.0 % - drop  9.7 % - 19414
            - cpap:11_cmh2o - 19381
            - respiratory event - central apnea - desat 80.0 % - 19304
            - respiratory event - obstructive apnea - desat 72.0 % - 19300
            - * desaturation - min 82.0 % - drop 13.7 % - 19077
            - * desaturation - min 85.0 % - drop  7.6 % - 18927
            - position - prone - 1 - 18779
            - * desaturation - min 79.0 % - drop 19.4 % - 18667
            - * desaturation - min 82.0 % - drop 15.5 % - 18602
            - * desaturation - min 86.0 % - drop  5.5 % - 18600
            - * desaturation - min 76.0 % - drop 21.6 % - 18511
            - * bilevel - applied - ipap   8.0 cmh2o - epap   8.0 cmh2o - 18464
            - * desaturation - min 79.0 % - drop 18.6 % - 18443
            - * desaturation - min 86.0 % - drop  4.4 % - 18322
            - respiratory event - mixed apnea - desat 77.0 % - 18253
            - new montage - 1c) standard montage with asv and etco2 - 18223
            - montage:01_standard_montage_aasm_[02] - 18102
            - montage:01_standard_montage_aasm_ref - 17949
            - new sensitivity 5000 µvp-p airflow - 17777
            - obstructive_hypopnea - 17713
            - montage:01_standard_montage_aasm_[04] - 17651
            - * desaturation - min 83.0 % - drop 11.7 % - 17174
            - new sensitivity 2000µv abd - 17107
            - * desaturation - min 79.0 % - drop 16.8 % - 16978
            - montage:01_standard_montage_aasm - 16953
            - cpap:12_cmh2o - 16944
            - * desaturation - min 85.0 % - drop  8.6 % - 16725
            - * desaturation - min 81.0 % - drop 14.7 % - 16702
            - * desaturation - min 81.0 % - drop 16.5 % - 16623
            - * desaturation - min 80.0 % - drop 17.5 % - 16472
            - respiratory event - central apnea - desat 78.0 % - 16365
            - respiratory event - obstructive apnea - desat 71.0 % - 16320
            - * desaturation - min sao2 95.0 % - 15987
            - * desaturation - min 78.0 % - drop 19.6 % - 15881
            - respiratory event - hypopnea - desat 75.0 % - 15775
            - * desaturation - min 77.0 % - drop 20.6 % - 15728
            - cpap:_off - 15708
            - pulse_rate_event:_no_sensor - 15548
            - * desaturation - min 79.0 % - drop 17.7 % - 15519
            - bilevel - applied - ipap  14.0 cmh2o - epap  14.0 cmh2o - 15472
            - position - disconnect - 15457
            - * desaturation - min 78.0 % - drop 20.4 % - 15435
            - new sensitivity 750µv abd - 15355
            - rem-supine - 15110
            - * desaturation - min 77.0 % - drop 21.4 % - 15099
            - hypopneas - 15071
            - * desaturation - min 83.0 % - drop 10.8 % - 14972
            - new sensitivity 2000µv chest - 14963
            - ekg - 14903
            - new sensitivity 10000 µvp-p airflow - 14843
            - pulse_rate_event:_sensor_off - 14688
            - respiratory event - central apnea - desat 77.0 % - 14555
            - * desaturation - min 80.0 % - drop 15.8 % - 14414
            - new sensitivity 500 µvp-p chest - 14339
            - * bilevel - applied - ipap   9.0 cmh2o - epap   9.0 cmh2o - 14262
            - respiratory event - rera - desat 87.0 % - 14229
            - * desaturation - min 90.0 % - drop  9.1 % - 14125
            - reras - 14020
            - coughing - 14007
            - * desaturation - min 75.0 % - drop 21.9 % - 13960
            - * desaturation - min 78.0 % - drop 17.9 % - 13960
            - * desaturation - min 78.0 % - drop 18.8 % - 13907
            - * plm - periodic - 13361
            - new sensitivity 1500µv abd - 13000
            - plm - dur:  1.7 sec. - periodic - 12997
            - nasal breaths - 12928
            - * bilevel - applied - ipap   0.0 cmh2o - epap   0.0 cmh2o - 12830
            - * desaturation - min 76.0 % - drop 20.0 % - 12795
            - new sensitivity 750 µvp-p airflow - 12723
            - respiratory event - central apnea - desat 76.0 % - 12675
            - * desaturation - min 80.0 % - drop 18.4 % - 12617
            - * desaturation - min 77.0 % - drop 19.8 % - 12570
            - plm - dur:  1.2 sec. - periodic - 12354
            - respiratory event - hypopnea - desat 97.0 % - 12352
            - montage:01_standard_montage_aasm_[03] - 12325
            - * desaturation - min 75.0 % - drop 21.1 % - 12223
            - impedance_at_20_kohm - 12169
            - respiratory event - mixed apnea - desat 82.0 % - 12076
            - body_position:_prone - 12043
            - cpap:13_cmh2o - 11866
            - * desaturation - min 84.0 % - drop  8.7 % - 11860
            - new sensitivity 1500µv airflow - 11798
            - * desaturation - min 76.0 % - drop 20.8 % - 11790
            - * desaturation - min 76.0 % - drop 22.4 % - 11748
            - bathroom_visit - 11551
            - * desaturation - min 76.0 % - drop 18.3 % - 11432
            - * desaturation - min 81.0 % - drop 13.8 % - 11406
            - respiratory event - mixed apnea - desat 74.0 % - 11397
            - new sensitivity 500µv abd - 11268
            - * desaturation - min 74.0 % - drop 22.1 % - 11263
            - respiratory event - hypopnea - desat 74.0 % - 11242
            - plm - dur:  1.6 sec. - periodic - 11237
            - plm - resp obstructive - 11201
            - respiratory event - mixed apnea - desat 75.0 % - 11125
            - hob flat 2 pillows - 11010
            - new sensitivity 3000µv chest - 11007
            - spo2_event:_no_sensor - 10884
            - plm - dur:  1.1 sec. - periodic - 10831
            - new montage - 1b) standard montage with cpap and etco2 - 10812
            - * desaturation - min 75.0 % - drop 20.2 % - 10677
            - * desaturation - min 78.0 % - drop 17.0 % - 10663
            - respiratory event - mixed apnea - desat 95.0 % - 10638
            - * desaturation - min 76.0 % - drop 19.1 % - 10609
            - bilevel - applied - ipap  12.0 cmh2o - epap   8.0 cmh2o - 10608
            - respiratory event - mixed apnea - desat 80.0 % - 10453
            - * desaturation - min 82.0 % - drop 11.8 % - 10441
            - * bilevel - applied - ipap   7.0 cmh2o - epap   7.0 cmh2o - 10432
            - respiratory event - mixed apnea - desat 92.0 % - 10402
            - respiratory event - mixed apnea - desat 88.0 % - 10383
            - lm - isolated - 10354
            - plm - dur:  1.4 sec. - periodic - 10268
            - new sensitivity 200µv airflow - 10251
            - new sensitivity 5000µv abd - 10230
            - respiratory event - mixed apnea - desat 89.0 % - 10174
            - respiratory event - mixed apnea - desat 81.0 % - 10138
            - new sensitivity 200µv chest - 9995
            - * desaturation - min 82.0 % - drop 12.8 % - 9989
            - new sensitivity 750 µvp-p abd - 9987
            - * desaturation - min 85.0 % - drop  6.6 % - 9976
            - respiratory event - mixed apnea - desat 86.0 % - 9961
            - waveform polarity changed to inverted: ptaf - 9814
            - respiratory event - mixed apnea - desat 91.0 % - 9809
            - position - sitting - 1 - 9784
            - plm - dur:  1.3 sec. - periodic - 9742
            - new sensitivity 100 µvp-p chin1-chin2 - 9605
            - respiratory event - mixed apnea - desat 84.0 % - 9559
            - stage - no stage - 9525
            - position - sitting - 9498
            - new sensitivity 4000µv abd - 9454
            - * desaturation - min 74.0 % - drop 22.9 % - 9451
            - desaturation - min 93.0 % - drop  4.0 % - 9402
            - respiratory event - mixed apnea - desat 90.0 % - 9344
            - new sensitivity 150µv airflow - 9332
            - dc channel number display turned off: ptaf - 9139
            - * desaturation - min 77.0 % - drop 18.1 % - 9069
            - patient_having_events_raised_pressure - 9014
            - * bilevel - applied - ipap   4.0 cmh2o - epap   4.0 cmh2o - 8971
            - plm - dur:  2.0 sec. - periodic - 8866
            - respiratory event - rera - desat 86.0 % - 8809
            - * desaturation - min 79.0 % - drop 16.0 % - 8808
            - new sensitivity 500 µvp-p airflow - 8757
            - respiratory event - hypopnea - desat 73.0 % - 8694
            - new sensitivity 750 µvp-p ekg - 8689
            - bilevel - applied - ipap  15.0 cmh2o - epap  15.0 cmh2o - 8672
            - * desaturation - min 74.0 % - drop 21.3 % - 8665
            - * desaturation - min 89.0 % - drop 10.1 % - 8657
            - bilevel - applied - ipap  10.0 cmh2o - epap   6.0 cmh2o - 8651
            - * desaturation - min 77.0 % - drop 18.9 % - 8595
            - respiratory event - obstructive apnea - desat 99.0 % - 8574
            - montage:1_standard_montage - 8496
            - desaturation - min 93.0 % - drop -4.0 % - 8471
            - plm - dur:  0.8 sec. - periodic - 8420
            - * desaturation - min 85.0 % - drop  4.5 % - 8390
            - * desaturation - min 77.0 % - drop 16.3 % - 8344
            - new sensitivity 4000µv chest - 8280
            - bilevel - applied - ipap  14.0 cmh2o - epap  10.0 cmh2o - 8242
            - talking - 8227
            - respiratory event - central apnea - desat 75.0 % - 8211
            - * plm episode - 8189
            - * plm - isolated - 8189
            - new sensitivity 100 µvp-p snore - 8185
            - * desaturation - min 78.0 % - drop 16.1 % - 8169
            - impedance_at_5_kohm - 8152
            - cpap:14_cmh2o - 8142
            - apneas / desats - 8128
            - new sensitivity 3000µv abd - 8068
            - * desaturation - min 79.0 % - drop 14.1 % - 8036
            - desaturation - min 92.0 % - drop  4.0 % - 7997
            - * desaturation - min 83.0 % - drop  9.8 % - 7824
            - bed flat ; 1 pillow - 7779
            - respiratory event - mixed apnea - desat 87.0 % - 7762
            - desaturation - min 90.0 % - drop  4.0 % - 7738
            - montage:1_standard_montage_ref - 7727
            - * bilevel - applied - ipap  10.0 cmh2o - epap  10.0 cmh2o - 7717
            - pulse_rate_event:_low_perfusion - 7709
            - cont supine - 7629
            - respiratory event - obstructive apnea - desat 67.0 % - 7600
            - new sensitivity 150 µvp-p snore - 7598
            - * desaturation - min 84.0 % - drop  7.7 % - 7586
            - * desaturation - min 77.0 % - drop 17.2 % - 7495
            - analyzer_montage_changed_-_sleep_events - 7487
            - increased cpap for obstruction - 7418
            - new sensitivity 2000µv airflow - 7414
            - * desaturation - min 80.0 % - drop 14.0 % - 7386
            - bilevel - applied - ipap  15.0 cmh2o - epap  11.0 cmh2o - 7380
            - plm - dur:  2.3 sec. - periodic - 7380
            - * desaturation - min 79.0 % - drop 15.1 % - 7372
            - * desaturation - min 83.0 % - drop  5.7 % - 7369
            - respiratory event - hypopnea - desat 72.0 % - 7309
            - * desaturation - min 81.0 % - drop 12.9 % - 7286
            - * desaturation - min 80.0 % - drop 14.9 % - 7271
            - * desaturation - min 76.0 % - drop 17.4 % - 7260
            - new sensitivity 3000 µvp-p 4000 µvp-p - 7235
            - * desaturation - min 94.0 % - drop  3.1 % - 7205
            - desaturation - min 92.0 % - drop -4.0 % - 7117
            - new sensitivity 2000 µvp-p ekg - 7094
            - waveform polarity changed to inverted: cflow - 7072
            - respiratory event - mixed apnea - desat 85.0 % - 7069
            - * desaturation - min 93.0 % - drop  3.1 % - 7022
            - respiratory event - mixed apnea - desat 94.0 % - 7014
            - desaturation - min 91.0 % - drop  4.0 % - 6917
            - * desaturation - min 75.0 % - drop 22.7 % - 6910
            - waveform polarity changed to normal: airflow - 6889
            - plm - dur:  1.9 sec. - isolated - 6886
            - bilevel - applied - ipap  11.0 cmh2o - epap   7.0 cmh2o - 6863
            - plm - resp rera - 6818
            - respiratory event - mixed apnea - desat 83.0 % - 6803
            - * desaturation - min 80.0 % - drop 13.0 % - 6801
            - new sensitivity 5000µv chest - 6773
            - new sensitivity 150µv chest - 6746
            - * bilevel - applied - ipap   5.0 cmh2o - epap   5.0 cmh2o - 6736
            - * desaturation - min 81.0 % - drop 12.0 % - 6729
            - new montage - 2a) standard average montage with cpap - 6722
            - * bilevel - applied - ipap   6.0 cmh2o - epap   6.0 cmh2o - 6715
            - * desaturation - min 80.0 % - drop 12.1 % - 6697
            - * bilevel - applied - ipap   3.0 cmh2o - epap   3.0 cmh2o - 6683
            - new sensitivity 200µv snore - 6683
            - new sensitivity 75 µvp-p chin1-chin2 - 6653
            - * desaturation - min 84.0 % - drop  4.5 % - 6644
            - * desaturation - min 75.0 % - drop 23.5 % - 6619
            - * desaturation - min 82.0 % - drop  8.9 % - 6597
            - respiratory event - obstructive apnea - desat 70.0 % - 6597
            - respiratory event - obstructive apnea - desat 65.0 % - 6595
            - plm - dur:  0.1 sec. - periodic - 6532
            - desaturation - min 89.0 % - drop  4.0 % - 6529
            - new sensitivity 100µv snore - 6528
            - waveform polarity changed to normal: ekg - 6527
            - patient not sleeping - 6513
            - new sensitivity 2000 µvp-p 3000 µvp-p - 6504
            - new high filters 35.0hz notch 60hz ekg - 6485
            - desaturation - min 91.0 % - drop -4.0 % - 6475
            - video_recording_off - 6467
            - * desaturation - min 75.0 % - drop 18.5 % - 6464
            - * desaturation - min 78.0 % - drop 15.2 % - 6437
            - * desaturation - min 76.0 % - drop 16.5 % - 6369
            - new sensitivity 300µv abd - 6364
            - new sensitivity 3000µv airflow - 6346
            - respiratory event - rera - desat 85.0 % - 6330
            - * snore - isolated - 6304
            - bed flat ; 2 pillows - 6284
            - * desaturation - min 74.0 % - drop 23.7 % - 6272
            - plm - dur:  2.1 sec. - periodic - 6270
            - leg movements - 6247
            - new low filters  1.0hz notch  ekg new high filters 70.0hz notch ekg - 6208
            - * desaturation - min 84.0 % - drop  5.6 % - 6201
            - * bilevel - applied - ipap  12.0 cmh2o - epap  12.0 cmh2o - 6185
            - plm - dur:  1.9 sec. - periodic - 6040
            - new sensitivity 75 µvp-p snore - 5969
            - plm - dur:  2.7 sec. - periodic - 5946
            - * desaturation - min 75.0 % - drop 19.4 % - 5943
            - new sensitivity 4000 µvp-p 5000 µvp-p - 5905
            - desaturation - min 91.0 % - drop -5.0 % - 5876
            - * desaturation - min 88.0 % - drop 11.1 % - 5866
            - respiratory event - obstructive apnea - desat 69.0 % - 5855
            - plm - dur:  1.2 sec. - isolated - 5804
            - * desaturation - min 85.0 % - drop  5.6 % - 5800
            - plm - dur:  2.5 sec. - periodic - 5791
            - new sensitivity µv chest - 5736
            - plm - dur:  1.5 sec. - periodic - 5698
            - post arousal - 5679
            - * desaturation - min sao2 78.0 % - 5644
            - new high filters 50.0hz notch 60hz ekg - 5547
            - patient movement - 5541
            - respiratory event - obstructive apnea - desat 66.0 % - 5528
            - new sensitivity 2000 µvp-p 4000 µvp-p - 5487
            - cpap increased for apneas / desats - 5476
            - arousal - snore - 1 - 5456
            - request supine - 5443
            - respiratory event - mixed apnea - desat 93.0 % - 5416
            - plm - dur:  0.7 sec. - periodic - 5394
            - respiratory event - obstructive apnea - desat 63.0 % - 5385
            - * desaturation - min 74.0 % - drop 19.6 % - 5359
            - request_supine - 5350
            - * desaturation - min 95.0 % - drop  3.1 % - 5347
            - bilevel - applied - ipap  16.0 cmh2o - epap  12.0 cmh2o - 5345
            - rem-lateral - 5311
            - respiratory event - hypopnea - desat 71.0 % - 5308
            - cpap:15_cmh2o - 5307
            - new sensitivity 100µv airflow - 5301
            - * desaturation - min 82.0 % - drop 10.9 % - 5274
            - plm - dur:  1.8 sec. - periodic - 5257
            - new sensitivity 1500 µvp-p 5000 µvp-p - 5257
            - new montage - 5) mslt/mwt montage - 5199
            - nasal breathing - 5189
            - montage:01_standard_montage_aasm_[02]_ref - 5184
            - plm - dur:  3.0 sec. - periodic - 5180
            - * bilevel - applied - ipap  11.0 cmh2o - epap  11.0 cmh2o - 5119
            - bilevel - applied - ipap   8.0 cmh2o - epap   4.0 cmh2o - 5098
            - * bilevel - applied - ipap   2.0 cmh2o - epap   2.0 cmh2o - 5069
            - tir pt to restroom - 5062
            - hob flat, pillows x2 - 5058
            - * bilevel - applied - ipap   1.0 cmh2o - epap   1.0 cmh2o - 5057
            - new sensitivity 4000 µvp-p - 5037
            - swallow - 5030
            - new sensitivity 1000µv chest ,abd - 5026
            - deep_breath - 5014
            - hob flat 1 pillow - 4918
            - * desaturation - min 72.0 % - drop 25.0 % - 4898
            - sleep_onset - 4893
            - respiratory event - mixed apnea - desat 73.0 % - 4893
            - * desaturation - min 79.0 % - drop 13.2 % - 4873
            - bilevel - applied - ipap  17.0 cmh2o - epap  13.0 cmh2o - 4857
            - pre-rem - 4834
            - body_position:_upright - 4830
            - * desaturation - min 92.0 % - drop  3.2 % - 4806
            - dc channel number display turned off: cflow - 4795
            - respiratory event - central apnea - desat 74.0 % - 4787
            - new sensitivity 4000 µvp-p 10000 µvp-p - 4745
            - bathroom visit - 4724
            - desaturation - min 88.0 % - drop  5.0 % - 4699
            - new sensitivity 50 µvp-p chin1-chin2 - 4689
            - * desaturation - min 84.0 % - drop  6.7 % - 4689
            - new sensitivity 3000 µvp-p 10000 µvp-p - 4687
            - * desaturation - min 83.0 % - drop  8.8 % - 4669
            - new sensitivity 500 µvp-p ekg - 4662
            - new sensitivity 1000 µvp-p 3000 µvp-p - 4589
            - bilevel - applied - ipap  13.0 cmh2o - epap   9.0 cmh2o - 4588
            - new sensitivity 1000 µvp-p 1500 µvp-p - 4587
            - heavy breathing - 4564
            - desaturation - min 90.0 % - drop  5.0 % - 4558
            - new sensitivity 100µv ic - 4554
            - arousal - bruxism - 1 - 4548
            - new sensitivity 200µv abd - 4543
            - bed flat; 1 pillow - 4526
            - new sensitivity 5000 µvp-p - 4514
            - * desaturation - min 82.0 % - drop  9.9 % - 4511
            - pressure increased because of apneas - 4480
            - lights out  - 4462
            - * desaturation - min 95.0 % - drop  5.0 % - 4446
            - respiratory event - rera - desat 80.0 % - 4444
            - * desaturation - min 81.0 % - drop  6.9 % - 4440
            - respiratory event - mixed apnea - desat 72.0 % - 4421
            - * desaturation - min 73.0 % - drop 24.0 % - 4415
            - new sensitivity 1500 µvp-p 4000 µvp-p - 4399
            - desaturation - min 88.0 % - drop  4.0 % - 4393
            - respiratory event - central apnea - desat 71.0 % - 4365
            - * desaturation - min 73.0 % - drop 15.1 % - 4358
            - * desaturation - min 82.0 % - drop  7.9 % - 4354
            - arrhythmia - 4354
            - desaturation - min 94.0 % - drop  4.0 % - 4319
            - bilevel - applied - ipap  16.0 cmh2o - epap  16.0 cmh2o - 4309
            - * desaturation - min 81.0 % - drop  9.0 % - 4287
            - new sensitivity 1000 µvp-p ekg - 4287
            - pt coughing - 4241
            - new montage - 1a) standard average montage - 4227
            - snore - periodic - 1 - 4203
            - new sensitivity 1000 µvp-p 4000 µvp-p - 4180
            - new sensitivity 500 µvp-p snore - 4146
            - respiratory event - mixed apnea - desat 96.0 % - 4140
            - * desaturation - min 81.0 % - drop  5.8 % - 4136
            - plm - dur:  0.5 sec. - periodic - 4111
            - new sensitivity 3000 µvp-p - 4107
            - no breathing - 4094
            - new montage - standard montage - 4092
            - new sensitivity 1000 µvp-p ic - 4087
            - respiratory event - obstructive apnea - desat 68.0 % - 4082
            - plm - dur:  0.8 sec. - isolated - 4070
            - new sensitivity 750 µvp-p ic - 4059
            - bilevel - applied - ipap   9.0 cmh2o - epap   5.0 cmh2o - 4042
            - new sensitivity 1500 µvp-p 2000 µvp-p - 4042
            - * desaturation - min 81.0 % - drop 11.0 % - 4018
            - * desaturation - min 72.0 % - drop 14.3 % - 4016
            - audible snore - 4012
            - arousal / leg movement - 4011
            - * desaturation - min 87.0 % - drop 12.1 % - 3976
            - * desaturation - min 75.0 % - drop 17.6 % - 3974
            - new sensitivity 3000 µvp-p ekg - 3954
            - bed flat; 2 pillows - 3953
            - respiratory event - mixed apnea - desat  0.0 % - 3949
            - * desaturation - min 77.0 % - drop 14.4 % - 3947
            - plm - dur:  2.6 sec. - periodic - 3928
            - plm - dur:  2.8 sec. - periodic - 3909
            - plm - dur:  2.4 sec. - periodic - 3894
            - * desaturation - min 74.0 % - drop 18.7 % - 3890
            - * desaturation - min 78.0 % - drop 12.4 % - 3876
            - desaturation - min 90.0 % - drop -4.0 % - 3864
            - desaturation - min 94.0 % - drop -4.0 % - 3825
            - * desaturation - min 86.0 % - drop 13.1 % - 3815
            - * desaturation - min 82.0 % - drop  5.7 % - 3800
            - * desaturation - min 94.0 % - drop  6.0 % - 3799
            - new sensitivity µv abd - 3795
            - plm - dur:  0.6 sec. - periodic - 3781
            - tech in to administer cpap - 3765
            - rem on side - 3759
            - desaturation - min 89.0 % - drop -5.0 % - 3752
            - desaturation - min 91.0 % - drop  5.0 % - 3748
            - new sensitivity 500 µvp-p ic - 3719
            - plm - dur:  2.0 sec. - isolated - 3705
            - * desaturation - min 82.0 % - drop  6.8 % - 3684
            - new sensitivity 1500 µvp-p - 3674
            - * desaturation - min 83.0 % - drop  4.6 % - 3672
            - respiratory event - hypopnea - desat 70.0 % - 3666
            - new sensitivity 50 µvp-p chin1-chin2, new high filters 35.0hz notch 60hz chin1-chin2 - 3652
            - plm - dur:  1.7 sec. - isolated - 3649
            - * desaturation - min sao2 77.0 % - 3631
            - mouth breathing - 3628
            - * desaturation - min 81.0 % - drop  8.0 % - 3621
            - new sensitivity 1500 µvp-p 3000 µvp-p - 3620
            - horizontal eye movements - 3615
            - vertical eye movements - 3615
            - respiratory event - rera - desat 84.0 % - 3586
            - pt back from bathroom - 3575
            - * desaturation - min sao2 76.0 % - 3570
            - rem supine transition - 3548
            - plm - dur:  2.9 sec. - periodic - 3544
            - new sensitivity 4000µv airflow - 3508
            - * desaturation - min 78.0 % - drop 13.3 % - 3492
            - * desaturation - min 77.0 % - drop 15.4 % - 3491
            - obstructive and desat events in rem-supine - 3444
            - plm - dur:  0.2 sec. - periodic - 3408
            - plm - dur:  3.4 sec. - periodic - 3402
            - apnea - 3363
            - new sensitivity 300 µvp-p chest - 3356
            - back from bathroom - 3340
            - new sensitivity 50µv ic - 3337
            - bilevel - applied - ipap  15.0 cmh2o - epap  10.0 cmh2o - 3315
            - new sensitivity 3000 µvp-p abd, new low filters  0.1hz notch 60hz abd, new high filters 12.0hz notch 60hz abd - 3312
            - new low filters  0.1hz notch 60hz chest, new high filters 12.0hz notch 60hz chest - 3308
            - sleep onset - 3304
            - new sensitivity 1000µv ekg - 3291
            - new sensitivity 1500 µvp-p ekg - 3279
            - rem, supine - 3275
            - desaturation - min 87.0 % - drop  5.0 % - 3274
            - bilevel - applied - ipap   5.0 cmh2o - epap   0.0 cmh2o - 3273
            - * desaturation - min 80.0 % - drop  8.0 % - 3265
            - new sensitivity 200µv rat - 3260
            - tachycardia - 3249
            - * desaturation - min 78.0 % - drop 14.3 % - 3249
            - started_analyzer_-_ecg - 3248
            - bilevel - applied - ipap   6.0 cmh2o - epap   0.0 cmh2o - 3233
            - * desaturation - min 74.0 % - drop 24.5 % - 3221
            - nose breathing - 3221
            - waveform polarity changed to normal: ptaf - 3198
            - rem left - 3186
            - new sensitivity 5000 µvp-p 10000 µvp-p - 3185
            - montage:1_standard_montage_[01] - 3178
            - * arousal - dur:  0.1 sec. - respiratory event - 3160
            - desaturation - min 89.0 % - drop  6.0 % - 3160
            - tech requested supine position change - 3160
            - * snore episode - 3152
            - edit data - start - 3139
            - desaturation - min 87.0 % - drop  4.0 % - 3123
            - respiratory event - hypopnea - desat 65.0 % - 3119
            - * desaturation - min 78.0 % - drop 11.4 % - 3113
            - respiratory event - central apnea - desat 73.0 % - 3083
            - new sensitivity 300 µvp-p ic - 3070
            - new montage - 7) 12 channel eeg montage - 3064
            - cpap_increased_for_apneas_/_desats - 3032
            - plm - dur:  0.9 sec. - periodic - 3012
            - bilevel - applied - ipap  18.0 cmh2o - epap  14.0 cmh2o - 3009
            - montage:03_mslt_mwt_ref - 3006
            - plm - dur:  3.2 sec. - periodic - 2981
            - plm - dur:  2.2 sec. - periodic - 2977
            - new sensitivity 50 µvp-p snore - 2955
            - new sensitivity 2000µv ekg - 2937
            - new sensitivity 500 µvp-p abd - 2934
            - * bilevel - applied - ipap  13.0 cmh2o - epap  13.0 cmh2o - 2902
            - arouse - 2901
            - rem right - 2895
            - recording resumed after the pt used the bathroom - 2892
            - montage:1_standard_montage_[02] - 2876
            - plm - dur:  0.4 sec. - periodic - 2875
            - * desaturation - min 73.0 % - drop 16.1 % - 2874
            - rem onset - supine - 2869
            - * desaturation - min 74.0 % - drop 20.4 % - 2860
            - * desaturation - min 83.0 % - drop  7.8 % - 2855
            - waveform polarity changed to normal: cflow - 2844
            - * desaturation - min 71.0 % - drop 18.4 % - 2842
            - plm - dur:  2.3 sec. - isolated - 2842
            - plm - dur:  1.1 sec. - isolated - 2831
            - new sensitivity 1000 µvp-p 10000 µvp-p - 2830
            - plm - dur:  1.6 sec. - isolated - 2823
            - new sensitivity 100µv abd - 2813
            - new sensitivity 200 µvp-p lat - 2810
            - * desaturation - min 72.0 % - drop 15.3 % - 2802
            - plm - dur:  2.9 sec. - isolated - 2795
            - * desaturation - min 80.0 % - drop  7.0 % - 2789
            - bilevel - applied - ipap   7.0 cmh2o - epap   0.0 cmh2o - 2784
            - snores - 2775
            - * desaturation - min 80.0 % - drop 10.1 % - 2771
            - hypopneas, reras - 2764
            - back from the bathroom visit - 2762
            - * desaturation - min 73.0 % - drop 22.3 % - 2750
            - respiratory event - hypopnea - desat 98.0 % - 2748
            - * arousal - dur:  3.0 sec. - snore - 2747
            - waveform polarity changed to inverted: abd new sensitivity 4000 µvp-p abd - 2726
            - * desaturation - min 71.0 % - drop 26.8 % - 2723
            - new sensitivity 500µv ekg - 2717
            - new sensitivity 200 µvp-p rat - 2716
            - respiratory event - mixed apnea - desat 98.0 % - 2716
            - snore - dur:  1.0 sec. - isolated - 2712
            - bathroom - 2712
            - * desaturation - min 80.0 % - drop 11.1 % - 2711
            - desaturation - min 86.0 % - drop  4.0 % - 2704
            - plm - dur:  2.5 sec. - isolated - 2699
            - tech in to check on abd belt - 2698
            - pt movement still supine - 2691
            - new sensitivity 1500 µvp-p 10000 µvp-p - 2672
            - * desaturation - min 96.0 % - drop  4.0 % - 2669
            - new sensitivity 5000µv airflow - 2666
            - new sensitivity µv ekg - 2648
            - rem related events - 2640
            - plm - dur:  3.8 sec. - periodic - 2630
            - bipap_increased_for_apneas_/_desats - 2628
            - * desaturation - min 83.0 % - drop  6.7 % - 2622
            - respiratory event - obstructive apnea - desat 61.0 % - 2613
            - new montage - 3) standard montage with asv and etco2 - 2610
            - plm - dur:  1.8 sec. - isolated - 2605
            - plm - dur:  5.6 sec. - periodic - 2600
            - waveform polarity changed to inverted: abd  - 2593
            - new montage - 9a) mslt/mwt montage - 2588
            - recording_analyzer_-_sleep_events - 2570
            - tech_requested_pt_turn_supine - 2486
            - new sensitivity 35 µvp-p chin1-chin2 - 2467
            - * desaturation - min sao2 67.0 % - 2460
            - * desaturation - min sao2 66.0 % - 2460
            - 2 pillows - 2457
            - new sensitivity 3000µv ekg - 2454
            - apnea / desats - 2444
            - new sensitivity µv airflow - 2441
            - new sensitivity 500µv ic - 2434
            - started_analyzer_-_data_trends - 2423
            - * desaturation - min 85.0 % - drop 14.1 % - 2420
            - new sensitivity 750 µvp-p 1000 µvp-p - 2410
            - plm - dur:  1.3 sec. - isolated - 2405
            - * desaturation - min 73.0 % - drop 25.5 % - 2401
            - desaturation - min 90.0 % - drop  7.0 % - 2398
            - hob flat, pillows x1 - 2363
            - adjusts_mask - 2360
            - desaturation - min 85.0 % - drop 10.0 % - 2356
            - * desaturation - min 77.0 % - drop 10.5 % - 2351
            - * desaturation - min 78.0 % - drop 10.3 % - 2348
            - new sensitivity 75µv airflow - 2343
            - new sensitivity 300µv snore - 2339
            - * desaturation - min 80.0 % - drop  5.9 % - 2337
            - new sensitivity 750µv ekg - 2336
            - desaturation - min 88.0 % - drop  7.0 % - 2331
            - desaturation - min 88.0 % - drop  6.0 % - 2329
            - new sensitivity 50µv chin1-chin2 - 2321
            - * desaturation - min 77.0 % - drop 12.5 % - 2321
            - desaturation - min 89.0 % - drop -6.0 % - 2320
            - * desaturation - min 72.0 % - drop 18.2 % - 2319
            - new sensitivity 3000 µvp-p 5000 µvp-p - 2304
            - waveform polarity changed to inverted: chest ,inverted: abd - 2304
            - * desaturation - min 71.0 % - drop 26.0 % - 2302
            - * desaturation - min 73.0 % - drop 23.2 % - 2298
            - * arousal - dur:  5.1 sec. - spontaneous - 2284
            - desaturation - min 90.0 % - drop  6.0 % - 2274
            - new sensitivity 1000 µvp-p - 2273
            - respiration resumed - 2268
            - oral respiration - 2267
            - nasal respiration - 2267
            - bilevel - applied - ipap   8.0 cmh2o - epap   0.0 cmh2o - 2267
            - desaturation - min 87.0 % - drop  6.0 % - 2261
            - new sensitivity 2000 µvp-p - 2257
            - * arousal - dur:  7.0 sec. - spontaneous - 2256
            - new sensitivity 1000 µvp-p 2000 µvp-p - 2255
            - new sensitivity 750µv chest ,1000µv abd - 2237
            - * desaturation - min 73.0 % - drop 24.7 % - 2236
            - pt_coughing - 2234
            - respiratory event - obstructive apnea - desat 60.0 % - 2230
            - new sensitivity 200µv lat - 2219
            - desaturation - min 92.0 % - drop  3.0 % - 2219
            - * desaturation - min 77.0 % - drop 13.5 % - 2219
            - * desaturation - min 81.0 % - drop 10.0 % - 2213
            - plm - dur:  3.5 sec. - periodic - 2212
            - post-arousal - 2202
            - respiratory event - mixed apnea - desat 70.0 % - 2196
            - * desaturation - min 79.0 % - drop  4.8 % - 2193
            - respiratory event - obstructive apnea - desat 62.0 % - 2193
            - respiratory event - obstructive apnea - desat 59.0 % - 2190
            - centrals, hypopneas - 2188
            - * desaturation - min 80.0 % - drop  9.1 % - 2187
            - respiratory event - central apnea - desat 72.0 % - 2180
            - respiratory event - central apnea - desat 100.0 % - 2167
            - * respiratory event - central apnea - desat 93.0 % - 2155
            - * desaturation - dur: 31.0 sec. - min 92.0 % - drop  6.1 % - 2130
            - bilevel - applied - ipap  19.0 cmh2o - epap  15.0 cmh2o - 2079
            - new montage - 4) 13 channel eeg montage - 2064
            - tech requested supine - 2058
            - new sensitivity 25µv chin1-chin2 - 2057
            - new sensitivity 200µv ic - 2046
            - respiratory event - rera - desat 83.0 % - 2041
            - patient back from the bathroom - 2029
            - this is my 1st pt to wake - 2026
            - * desaturation - min 91.0 % - drop  3.2 % - 2021
            - * desaturation - min 71.0 % - drop 13.4 % - 2020
            - new sensitivity 25µv chest - 2014
            - montage:02_standard_montage_with_etco2_[02] - 2012
            - * desaturation - min 74.0 % - drop  9.8 % - 1964
            - * desaturation - min 71.0 % - drop 15.5 % - 1963
            - increase  pressure for apnea - 1948
            - tech in to start treatment - 1947
            - plm - dur:  0.9 sec. - isolated - 1946
            - * desaturation - min 74.0 % - drop 10.8 % - 1946
            - new sensitivity 150µv abd - 1943
            - back from bathroom visit - 1938
            - new sensitivity 1500 µvp-p chest, new low filters  0.1hz notch 60hz chest, new high filters 12.0hz notch 60hz chest - 1937
            - * desaturation - min 72.0 % - drop 21.7 % - 1935
            - new sensitivity 50µv airflow - 1935
            - plm - dur:  3.1 sec. - periodic - 1931
            - new sensitivity 150 µvp-p - 1931
            - conts supine - 1925
            - * desaturation - min 79.0 % - drop  6.0 % - 1924
            - apneas - 1922
            - rera's, and hypopneas cpap up to 8cm h2o - 1922
            - periodic breathing - 1920
            - * desaturation - dur: 17.3 sec. - min 91.0 % - drop  4.2 % - 1915
            - montage:1_standard_montage_[03] - 1912
            - * desaturation - min 80.0 % - drop  4.8 % - 1910
            - new sensitivity 150µv snore - 1908
            - stage - mvt - 1908
            - respiratory event - hypopnea - desat 69.0 % - 1908
            - new sensitivity 300 µvp-p airflow - 1906
            - plm - dur:  2.2 sec. - isolated - 1906
            - pt aroused - 1904
            - recording resumed after redacted used the bathroom - 1900
            - pt reading - 1898
            - channel display turned off: airflow - 1898
            - remains supine - 1897
            - * desaturation - min 93.0 % - drop  7.0 % - 1892
            - chg chin montage - 1891
            - desaturation - min 92.0 % - drop  6.0 % - 1891
            - respiratory event - partial obstructive - desat 95.0 % - 1890
            - * desaturation - min 79.0 % - drop 10.2 % - 1890
            - new sensitivity 750 µvp-p 3000 µvp-p - 1889
            - new sensitivity 300 µvp-p snore - 1887
            - new sensitivity 150µv ic - 1885
            - desaturation - min 93.0 % - drop  5.0 % - 1882
            - new sensitivity 750 µvp-p 5000 µvp-p - 1882
            - hypopneas cpap up to 7cm h2o - 1878
            - pt did not stay supine - 1877
            - respiratory event - mixed apnea - desat 97.0 % - 1877
            - bradycardia - 1868
            - new sensitivity 500µv snore - 1868
            - new sensitivity 50 µvp-p chin1-chin2, new high filters 50.0hz notch 60hz chin1-chin2 - 1863
            - desaturation - min 89.0 % - drop  5.0 % - 1863
            - * arousal - dur:  6.1 sec. - spontaneous - 1860
            - * desaturation - min 73.0 % - drop 19.8 % - 1856
            - * desaturation - min 79.0 % - drop 12.2 % - 1855
            - new sensitivity µv snore - 1854
            - new sensitivity 1000 µvp-p chest, new low filters  0.1hz notch 60hz chest, new high filters 12.0hz notch 60hz chest - 1853
            - * desaturation - min 73.0 % - drop 17.0 % - 1851
            - desaturation - min 87.0 % - drop  8.0 % - 1851
            - * desaturation - min 71.0 % - drop 25.3 % - 1850
            - new sensitivity 200 µvp-p - 1849
            - respevent - partialobstructive - 5 - 1848
            - heavy_breathing - 1847
            - * desaturation - min 79.0 % - drop  8.1 % - 1847
            - partially supine - 1847
            - cpap increased for pt. comfort - 1847
            - cpap to 6 for desats - 1844
            - * desaturation - min sao2 75.0 % - 1843
            - * desaturation - min 72.0 % - drop 25.8 % - 1837
            - * desaturation - min 77.0 % - drop  8.3 % - 1836
            - partially prone - 1836
            - pt movement still left - 1833
            - * desaturation - dur: 60.0 sec. - min sao2 89.0 % - 1830
            - waveform polarity changed to normal: chest  - 1829
            - desaturation - min 90.0 % - drop  3.0 % - 1829
            - new sensitivity 100µv chin1-chin2 - 1826
            - new sensitivity 150 µvp-p chin1-chin2 - 1823
            - patient is back from the bathroom - 1822
            - desaturation - min 92.0 % - drop -6.0 % - 1822
            - desaturation - min 86.0 % - drop  8.0 % - 1819
            - new sensitivity 35µv airflow - 1818
            - * desaturation - min 71.0 % - drop 27.6 % - 1818
            - pt aroused tech requested supine position change - 1817
            - new sensitivity 1000 µvp-p ekg, new high filters 35.0hz notch 60hz ekg - 1817
            - desaturation - min 89.0 % - drop  3.0 % - 1817
            - * arousal - dur:  0.1 sec. - plm - 1815
            - new sensitivity 150µv rat - 1814
            - * desaturation - dur: 26.7 sec. - min 93.0 % - drop  5.1 % - 1813
            - desaturation - min 91.0 % - drop  3.0 % - 1812
            - waveform polarity changed to normal: chest ,normal: abd - 1811
            - cpap increased for apneas / desats / reras - 1810
            - respiratory event - central apnea - desat 69.0 % - 1810
            - respiratory event - central apnea - desat 68.0 % - 1810
            - respiratory event - central apnea - desat 70.0 % - 1810
            - new sensitivity 75µv chin1-chin2 - 1809
            - desaturation - min 93.0 % - drop  0.0 % - 1805
            - recording_analyzer_-_csa - 1801
            - * desaturation - min 75.0 % - drop 11.8 % - 1800
            - * desaturation - min 72.0 % - drop 24.2 % - 1794
            - bilevel - applied - ipap   9.0 cmh2o - epap   0.0 cmh2o - 1794
            - new sensitivity 10000 µvp-p abd, new low filters  0.1hz notch 60hz abd, new high filters 12.0hz notch 60hz abd - 1789
            - new sensitivity 10000 µvp-p - 1788
            - movement, moan, groans - 1788
            - * arousal - dur:  0.0 sec. - spontaneous - 1784
            - rem lateral - 1782
            - mod. snore - 1776
            - rem desats - 1776
            - tech out, pt needed bathroom - 1772
            - desaturation - min 86.0 % - drop  7.0 % - 1765
            - desaturation - min 92.0 % - drop -5.0 % - 1763
            - * desaturation - dur: 30.0 sec. - min 91.0 % - drop  6.2 % - 1762
            - * desaturation - dur: 36.3 sec. - min 91.0 % - drop  7.1 % - 1762
            - * desaturation - dur: 30.7 sec. - min 92.0 % - drop  5.2 % - 1762
            - * arousal - dur:  0.2 sec. - respiratory event - 1759
            - cpap:16_cmh2o - 1758
            - new sensitivity 200 µvp-p ekg - 1756
            - cpap increased for snoring - 1756
            - waveform polarity changed to normal: abd  - 1755
            - * arousal - dur:  4.6 sec. - spontaneous - 1753
            - new sensitivity 100µv chest - 1751
            - new sensitivity 150µv chin1-chin2 - 1740
            - new sensitivity 200 µvp-p ic - 1738
            - desaturation - min 88.0 % - drop -4.0 % - 1734
            - plm - dur:  3.3 sec. - periodic - 1734
            - * desaturation - min 70.0 % - drop 27.8 % - 1734
            - new sensitivity 500µv chest ,1500µv abd - 1729
            - adjusted mask for leak - 1729
            - respiratory event - rera - desat 82.0 % - 1727
            - plm - dur:  1.5 sec. - isolated - 1725
            - * respiratory event - obstructive apnea - desat 95.0 % - 1724
            - * respiratory event - central apnea - desat 94.0 % - 1724
            - * respiratory event - obstructive apnea - desat 91.0 % - 1724
            - * desaturation - min 75.0 % - drop 16.7 % - 1721
            - * arousal - dur:  6.2 sec. - spontaneous - 1708
            - new sensitivity 1500µv chest ,abd - 1707
            - * desaturation - dur: 32.3 sec. - min 92.0 % - drop  6.1 % - 1704
            - pt supine/to left - 1670
            - start_of_study - 1668
            - respiratory event - hypopnea - desat 68.0 % - 1641
            - respiratory event - obstructive apnea - desat 57.0 % - 1615
            - respiratory event - rera - desat 79.0 % - 1596
            - * desaturation - min sao2 72.0 % - 1593
            - new sensitivity 75µv snore - 1531
            - * desaturation - dur: 11.0 sec. - min sao2 91.0 % - 1525
            - new sensitivity 5000 µvp-p ic - 1524
            - * desaturation - min 90.0 % - drop  3.2 % - 1518
            - 5_quick_breaths - 1509
            - patient_bathroom - 1501
            - desaturation - min  0.0 % - drop  5.0 % - 1501
            - plm - dur:  0.7 sec. - isolated - 1499
            - * desaturation - min 65.0 % - drop 22.6 % - 1499
            - * desaturation - min 73.0 % - drop 20.7 % - 1496
            - patient_in_rem_supine_having_events_raised_pressure - 1494
            - * desaturation - min 79.0 % - drop  9.2 % - 1493
            - events, arousals - 1479
            - ekg  arrhythmia - 1476
            - respiratory event - mixed apnea - desat 67.0 % - 1476
            - desaturation - min 90.0 % - drop -8.0 % - 1470
            - new sensitivity 3000 µvp-p ekg, new high filters 35.0hz notch 60hz ekg - 1470
            - desaturation - min 96.0 % - drop  0.0 % - 1467
            - new sensitivity 500µv chest ,1000µv abd - 1465
            - talking in rem - 1461
            - plm - resp hypopnea - 1461
            - desaturation - min 86.0 % - drop  5.0 % - 1452
            - cs_breathing - 1452
            - * desaturation - min 72.0 % - drop  8.9 % - 1451
            - mom coughing - 1442
            - resmed medium airfit f20 full face mask - 1440
            - * desaturation - min 71.0 % - drop 22.0 % - 1440
            - tech requests supine - 1439
            - * desaturation - min 73.0 % - drop 12.0 % - 1438
            - * desaturation - dur: 34.0 sec. - min 89.0 % - drop  9.2 % - 1438
            - new sensitivity 1000 µvp-p 5000 µvp-p - 1437
            - new sensitivity 300µv ekg - 1436
            - * desaturation - min 70.0 % - drop 23.1 % - 1434
            - ekg arrythmia - 1434
            - montage:01_standard_montage_aasm_[03]_ref - 1433
            - * desaturation - dur: 18.7 sec. - min 92.0 % - drop  4.2 % - 1431
            - * desaturation - dur: 29.3 sec. - min 88.0 % - drop  9.3 % - 1431
            - tech requested pt turn supine - 1428
            - audible snore arousal - 1428
            - * bilevel - applied - ipap  14.0 cmh2o - epap  14.0 cmh2o - 1427
            - * arousal - dur:  8.5 sec. - spontaneous - 1426
            - * desaturation - min 73.0 % - drop 14.1 % - 1426
            - * desaturation - min 71.0 % - drop 12.3 % - 1426
            - pt wearing a resmed mirage fx nasal mask standard size - 1425
            - * desaturation - dur: 24.7 sec. - min 91.0 % - drop  5.2 % - 1423
            - * desaturation - min 75.0 % - drop 10.7 % - 1420
            - * desaturation - min 73.0 % - drop 18.9 % - 1420
            - * desaturation - min 73.0 % - drop 21.5 % - 1420
            - ecg - 1420
            - desaturation - min 84.0 % - drop  4.0 % - 1420
            - respiration ceased - 1419
            - tech asks pt to roll over his back/supine - 1419
            - respironics comfort select nasal size medium with warm humidification - 1418
            - plm - dur:  3.6 sec. - periodic - 1418
            - changed chin montage - 1417
            - trying bipap - 1417
            - * desaturation - min 69.0 % - drop 29.6 % - 1416
            - * desaturation - min 84.0 % - drop 15.2 % - 1414
            - desaturation - min 95.0 % - drop  0.0 % - 1413
            - new sensitivity µv chest ,µv abd - 1411
            - cpap to 5 for snoring - 1411
            - increased pressure to 7 apnea - 1411
            - plm - 1410
            - resmed_mirage_fx_nasal_mask_standard_size - 1410
            - desaturation - min 92.0 % - drop -3.0 % - 1410
            - bilevel - applied - ipap  12.0 cmh2o - epap   9.0 cmh2o - 1408
            - new sensitivity 500 µvp-p 3000 µvp-p - 1407
            - waveform polarity changed to inverted: chest  - 1407
            - desaturation - min 91.0 % - drop -6.0 % - 1405
            - cpap to 6 events - 1405
            - new sensitivity 75 µvp-p - 1404
            - spo2_event:_low_quality - 1404
            - desats - 1403
            - new sensitivity 4000 µvp-p abd, new low filters  0.1hz notch 60hz abd, new high filters 12.0hz notch 60hz abd - 1403
            - back_from_bathroom - 1403
            - * desaturation - min 76.0 % - drop 13.6 % - 1400
            - * desaturation - min 73.0 % - drop 13.1 % - 1399
            - * desaturation - min 73.0 % - drop 18.0 % - 1397
            - more supine than right - 1397
            - desaturation - min 97.0 % - drop  0.0 % - 1396
            - tir_pt_to_restroom - 1395
            - conts. supine - 1392
            - bilevel - applied - ipap  15.0 cmh2o - epap   9.0 cmh2o - 1391
            - desaturation - min 90.0 % - drop -7.0 % - 1390
            - moderate_snores - 1389
            - patient said cannot sleep supine - 1388
            - increased pressure to 9 apneas - 1388
            - new sensitivity 1000µv airflow ,chest ,abd - 1387
            - bilevel - applied - ipap  11.0 cmh2o - epap   6.0 cmh2o - 1384
            - desaturation - min 88.0 % - drop -7.0 % - 1383
            - * arousal - dur:  3.4 sec. - respiratory event - 1382
            - respiratory event - obstructive apnea - desat 64.0 % - 1380
            - bilevel - applied - ipap  13.0 cmh2o - epap   8.0 cmh2o - 1380
            - n2  audible snore - 1380
            - new sensitivity 200 µvp-p chin1-chin3 - 1380
            - increased pressure to 8 apnea - 1379
            - * desaturation - dur: 25.0 sec. - min 92.0 % - drop  4.2 % - 1379
            - * desaturation - dur: 23.3 sec. - min 90.0 % - drop  5.3 % - 1379
            - * desaturation - dur: 22.7 sec. - min 89.0 % - drop  6.3 % - 1379
            - respiratory event - dur: 18.3 sec. - hypopnea - desat 91.0 % - 1378
            - pt still supine - 1377
            - new sensitivity 500µv abd ,1000µv chest - 1377
            - plm - dur:  2.1 sec. - isolated - 1377
            - * desaturation - dur: 31.0 sec. - min 91.0 % - drop  6.2 % - 1376
            - montage:1_standard_montage_[01]_ref - 1374
            - soft_snores - 1374
            - new sensitivity 2000 µvp-p 5000 µvp-p - 1374
            - waveform polarity changed to inverted: abd new sensitivity 10000 µvp-p abd - 1373
            - * arousal - dur:  7.9 sec. - spontaneous - 1373
            - desaturation - min 85.0 % - drop  9.0 % - 1371
            - new sensitivity 1000 µvp-p ekg, new high filters 50.0hz notch 60hz ekg - 1368
            - 4% desaturation - 1367
            - plm - dur:  4.2 sec. - periodic - 1365
            - * desaturation - dur: 26.7 sec. - min 93.0 % - drop  4.1 % - 1364
            - pt aroused, request supine - 1363
            - halfway supine - 1363
            - new sensitivity 750µv chest ,abd - 1362
            - * arousal - dur:  5.4 sec. - spontaneous - 1359
            - waveform polarity changed to normal: abd new sensitivity 10000 µvp-p abd - 1359
            - new sensitivity 50µv abd - 1358
            - * desaturation - min 76.0 % - drop 15.6 % - 1353
            - new sensitivity 50 µvp-p chin1-chin3 - 1353
            - * desaturation - min 77.0 % - drop 11.5 % - 1351
            - * desaturation - min sao2 74.0 % - 1351
            - increase for apnea - 1350
            - * desaturation - min 75.0 % - drop 14.8 % - 1350
            - cpap increased for reras - 1347
            - oral breathing - 1347
            - * desaturation - min 76.0 % - drop  6.2 % - 1346
            - * desaturation - min 65.0 % - drop 26.1 % - 1346
            - * arousal - dur:  4.6 sec. - respiratory event - 1341
            - * arousal - dur:  9.9 sec. - spontaneous - 1341
            - * desaturation - min 72.0 % - drop 26.5 % - 1339
            - respiratory event - dur: 10.0 sec. - central apnea - desat 95.0 % - 1338
            - hypopneas and rera's, cpap to 6cm h2o - 1336
            - * desaturation - min 77.0 % - drop  7.2 % - 1335
            - respiratory event - dur: 20.5 sec. - hypopnea - desat 89.0 % - 1334
            - respiratory event - mixed apnea - desat 71.0 % - 1330
            - * desaturation - dur: 34.7 sec. - min 91.0 % - drop  7.1 % - 1329
            - * desaturation - dur: 29.0 sec. - min 92.0 % - drop  6.1 % - 1329
            - supine rem - 1326
            - asv 5/15/0 - 1326
            - new sensitivity 150 µvp-p rat - 1324
            - desaturation - min 88.0 % - drop  8.0 % - 1324
            - cpap to 7 for snoring - 1322
            - plm - dur:  6.8 sec. - periodic - 1322
            - montage:new_01_standard_montage_aasm - 1320
            - cpap increased for apnea - 1320
            - leg movement / arousal - 1319
            - * arousal - dur:  6.6 sec. - respiratory event - 1318
            - new sensitivity 200 µvp-p snore - 1317
            - * arousal - dur: 11.7 sec. - respiratory event - 1317
            - * desaturation - min 72.0 % - drop 20.9 % - 1315
            - * desaturation - min 71.0 % - drop 17.4 % - 1315
            - * desaturation - min 70.0 % - drop 16.7 % - 1315
            - * arousal - dur:  3.4 sec. - spontaneous - 1314
            - * desaturation - min 78.0 % - drop  7.1 % - 1314
            - * arousal - dur:  7.4 sec. - spontaneous - 1314
            - * arousal - dur:  9.4 sec. - spontaneous - 1314
            - epap_increased_for_apneas_/_desats - 1314
            - * desaturation - min 72.0 % - drop 23.4 % - 1313
            - 1 pillow - 1311
            - * arousal - dur:  7.2 sec. - spontaneous - 1311
            - plm - dur:  5.7 sec. - periodic - 1308
            - new sensitivity 750 µvp-p - 1305
            - plm - dur:  4.7 sec. - periodic - 1305
            - waveform color changed snore new low filters 10.0hz  snore, new high filters 70.0hz  snore - 1304
            - * desaturation - min 68.0 % - drop 27.7 % - 1304
            - loud snores - 1302
            - * desaturation - min 69.0 % - drop 28.9 % - 1302
            - tightened mask - 1301
            - patient in rem having events raised press - 1300
            - new sensitivity 750 µvp-p ekg, new high filters 50.0hz notch 60hz ekg - 1299
            - respironics comfort select nasal mask size medium with warm humidification - 1297
            - cpap to 6 for snoring - 1297
            - * desaturation - min 68.0 % - drop 29.9 % - 1297
            - central_hypopnea - 1294
            - waveform polarity changed to inverted: chest new sensitivity 1500 µvp-p chest - 1294
            - desaturation - min 92.0 % - drop -2.0 % - 1294
            - * respiratory event - mixed apnea - desat 94.0 % - 1293
            - new sensitivity 750µv snore - 1292
            - respiratory event - obstructive apnea - desat 55.0 % - 1290
            - hand movements - 1287
            - desaturation - min 90.0 % - drop -5.0 % - 1284
            - * desaturation - min 69.0 % - drop 25.0 % - 1283
            - sinus arrhythmia - 1280
            - * desaturation - min 76.0 % - drop 10.6 % - 1278
            - * desaturation - dur: 28.7 sec. - min 93.0 % - drop  5.1 % - 1278
            - * desaturation - dur: 33.0 sec. - min 91.0 % - drop  7.1 % - 1278
            - * desaturation - dur: 34.0 sec. - min 91.0 % - drop  7.1 % - 1278
            - * desaturation - dur: 30.0 sec. - min 92.0 % - drop  6.1 % - 1278
            - * desaturation - dur: 35.0 sec. - min 91.0 % - drop  7.1 % - 1278
            - * desaturation - dur: 28.7 sec. - min 91.0 % - drop  6.2 % - 1278
            - * desaturation - dur: 34.3 sec. - min 92.0 % - drop  6.1 % - 1278
            - * desaturation - min 67.0 % - drop 28.0 % - 1276
            - plm - dur:  0.1 sec. - isolated - 1275
            - desaturation - min 87.0 % - drop -8.0 % - 1273
            - * arousal - dur:  0.1 sec. - spontaneous - 1266
            - * desaturation - min 78.0 % - drop  4.9 % - 1266
            - hypopneas_increase_cpap - 1266
            - cpap to 8 for desats - 1263
            - snore_arousal - 1260
            - * arousal - dur:  6.5 sec. - spontaneous - 1258
            - respiratory event - hypopnea - desat 67.0 % - 1254
            - patient having difficulty sleeping - 1254
            - patient unable to sleep in any position - 1254
            - desaturation - min 93.0 % - drop -5.0 % - 1248
            - cpap increased for reras / rem - 1242
            - hob_flat_2_pillows - 1240
            - patient_does_not_usually_sleep_supine - 1235
            - desaturation - min 90.0 % - drop -6.0 % - 1234
            - montage:3_mslt_mwt_ref - 1232
            - bradycardia 40's - 1232
            - increased_pressure_to_5_for_more_air - 1226
            - * desaturation - dur:  9.0 sec. - min sao2 90.0 % - 1220
            - snore - dur:  0.2 sec. - isolated - 1220
            - * arousal - dur:  1.9 sec. - respiratory event - 1220
            - * desaturation - dur: 10.0 sec. - min sao2 93.0 % - 1220
            - * desaturation - dur: 11.0 sec. - min sao2 93.0 % - 1220
            - * desaturation - dur:  8.0 sec. - min sao2 93.0 % - 1220
            - * arousal - dur: 11.4 sec. - spontaneous - 1211
            - apneas / arousals - 1209
            - respiratory event - hypopnea - desat 62.0 % - 1204
            - * desaturation - min 76.0 % - drop  8.4 % - 1200
            - * arousal - dur:  5.0 sec. - respiratory event - 1197
            - * arousal - dur:  4.1 sec. - respiratory event - 1194
            - patient does not usually sleep supine. tech explained  the signifigance of supine sleep for study. - 1194
            - plm - dur:  4.4 sec. - periodic - 1184
            - * arousal - dur:  3.8 sec. - respiratory event - 1156
            - * arousal - dur:  3.9 sec. - respiratory event - 1155
            - * arousal - dur:  4.3 sec. - spontaneous - 1152
            - * arousal - dur:  8.1 sec. - spontaneous - 1149
            - respiratory event - dur: 27.8 sec. - hypopnea - desat 90.0 % - 1143
            - new sensitivity 500µv airflow  - 1139
            - plm - dur:  4.5 sec. - periodic - 1129
            - desaturation - min sao2 88.0 % - 1096
            - * arousal - dur:  3.5 sec. - respiratory event - 1094
            - plm - dur:  0.6 sec. - isolated - 1079
            - * arousal - dur:  2.9 sec. - respiratory event - 1077
            - pressure increased, apneas - 1060
            - patient having difficulty sleeping supine - 1044
            - * arousal - dur:  3.6 sec. - respiratory event - 1036
            - new sensitivity 1000 µvp-p abd, new low filters  0.1hz notch 60hz abd, new high filters 12.0hz notch 60hz abd - 1035
            - fixing ekg - 1033
            - cpap up to 5cm h2o to encourage sleep - 1032
            - hypopneas and rera's, cpap up to 6cm h2o - 1032
            - awake supine and opening his mouth - 1032
            - rera's, will try 7cm h2o - 1032
            - rera's, and hypopneas cpap to 9cm h2o - 1032
            - patient having events in rem raised pres - 1032
            - obstructions, hypopneas cpap to 7cm h2o - 1024
            - cont'd resp events, hypop's, rera's, obstruc cpap to 8cm h2o - 1024
            - had chest and abdomen polarity mixed up - 1024
            - pt awake down on cpap - 1024
            - previous pressure for obstruc, hypopneas and rera's, 7cm h2o - 1024
            - previous pressure for obstruc, hypopneas and rera's, 8cm h2o - 1024
            - rera's, cpap to 9cmh2o - 1024
            - increased_pressure_apneas - 1022
            - new sensitivity 300µv ic - 1021
            - ekg- bigeminy-pvc's - 1020
            - desaturation - min 85.0 % - drop  5.0 % - 1019
            - new sensitivity 200 µvp-p chin1-chin2 - 1019
            - new sensitivity 750µv chest ,2000µv abd - 1018
            - new sensitivity 1000µv ic - 1016
            - cpap_8_cm_for_hypopneas - 1015
            - cpap_9_cm_for_hypopneas - 1015
            - position_changed - 1014
            - desaturation - min 86.0 % - drop  9.0 % - 1014
            - * desaturation - min 89.0 % - drop  3.3 % - 1013
            - * desaturation - min 88.0 % - drop  3.3 % - 1012
            - tech req supine - 1011
            - new sensitivity 300µv chest ,1000µv abd - 1011
            - * desaturation - min 81.0 % - drop  4.7 % - 1010
            - new sensitivity 500 µvp-p 2000 µvp-p - 1008
            - new sensitivity 2000 µvp-p rat - 1007
            - * desaturation - min 60.0 % - drop 35.5 % - 1007
            - * desaturation - min 78.0 % - drop  9.3 % - 1007
            - waveform color changed snore - 1007
            - res med swift fx nasal pillows, medium - 1006
            - patient_having_difficulty_sleeping - 1006
            - new sensitivity 4000µv ekg - 1006
            - one pillow used bed not elevated. - 1006
            - increased pressure for events - 1006
            - new sensitivity 4000 µvp-p chest, new low filters  0.1hz notch 60hz chest, new high filters 12.0hz notch 60hz chest - 1004
            - * desaturation - min 72.0 % - drop 17.2 % - 1004
            - * desaturation - min 72.0 % - drop 11.1 % - 1004
            - * desaturation - min 72.0 % - drop 13.3 % - 1004
            - * desaturation - min 74.0 % - drop 16.9 % - 1004
            - desaturation - min 89.0 % - drop -4.0 % - 1003
            - new sensitivity 150 µvp-p e1-m2 - 1002
            - new sensitivity 50 µvp-p chin1-chin3, new high filters 50.0hz notch 60hz chin1-chin3 - 1001
    '''
    def __init__(self, sane_labels=True, **kwargs):
        # HSP saturation samples are percentages although their EDF headers
        # incorrectly label these channels as microvolts.
        overrides = {"SaO2": "%", "SpO2": "%", "SPO2": "%"}
        overrides.update(kwargs.pop("edf_unit_overrides", {}) or {})
        super().__init__(edf_unit_overrides=overrides, **kwargs)
        self.sane_labels = sane_labels

    def get_event_df(self, edf_path, start_datetime):
        annot_path = get_hsp_annotation_path(edf_path)
        if annot_path is None:
            raise RuntimeError('No annotation file found for EDF file', basename(edf_path))

        df = pd.read_csv(annot_path)
        df = df.rename(columns={'event': 'Label', 'time': 'Starttime', 'duration': 'Duration'})
        df['Label'] = df['Label'].str.lower()
        if self.sane_labels:
            df = self._map_to_sane_labels(df)

        base_date = f'{start_datetime._date_repr}'
        df['Starttime'] = pd.to_datetime(base_date + ' ' + df['Starttime'])

        # Any time earlier than the first timestamp has rolled over to the next day
        # Add 1 day wherever the time is less than the starting time
        rollover_mask = df['Starttime'] < df['Starttime'].iloc[0]
        df.loc[rollover_mask, 'Starttime'] += pd.Timedelta(days=1)

        df['Endtime'] = df['Starttime'] + pd.to_timedelta(df['Duration'], unit='s')

        return df[['Label', 'Starttime', 'Endtime', 'Duration']]

    def _map_to_sane_labels(self, df):
        return map_hsp_sane_labels(df)
