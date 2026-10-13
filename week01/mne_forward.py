from pathlib import Path

import mne

data_path = Path(mne.datasets.sample.data_path())
subjects_dir = data_path / "subjects"
meg_path = data_path / "MEG" / "sample"

src = mne.setup_source_space("sample", spacing="oct6",
                             subjects_dir=subjects_dir, add_dist=False)
model = mne.make_bem_model("sample", ico=4, conductivity=(0.3,),  # 1-layer: MEG 전용
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
raw = mne.io.read_raw_fif(meg_path / "sample_audvis_raw.fif", preload=False)
info = raw.info

fwd_own = mne.make_forward_solution(
    info, trans=meg_path / "sample_audvis_raw-trans.fif",
    src=src, bem=bem, meg=True, eeg=False, mindist=5.0)