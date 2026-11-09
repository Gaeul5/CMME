"""Week 5-5 준비: 평가 머리(head012~017)마다 '후보 전도도별 리드필드'와 '전극 위치 오차별 정답 리드필드'를 만든다.

- 후보 전도도 리드필드: 두개골 전도도 0.004~0.015 중 6개 값, 전극은 모델 위치(오차 없음).
  1분 기록(라벨 없음)으로 전도도를 고르는 적응 실험(adapt.py E4)에 사용
- 전극 위치 오차별 정답 리드필드: 그 머리의 진짜 전도도, 전극 위치 오차 0/5/10/15 mm.
  자동 정합 오차가 결과에 미치는 영향(adapt.py E5)에 사용. 오차 방향은 머리마다 고정하고 크기만 바꿈

실행: python week05/heads_variants.py <MNE-sample-data 경로> <시작 머리> <끝 머리(포함 안 함)>
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heads as H  # noqa: E402

CANDS = [0.004, 0.006, 0.008, 0.010, 0.012, 0.015]
JITTERS = [0.0, 5.0, 10.0, 15.0]
OUT = os.path.join(H.HERE, "cache_variants")


def main(data_path, h0, h1):
    os.makedirs(OUT, exist_ok=True)
    P = H.paths(data_path)
    z0 = np.load(os.path.join(H.CACHE0, "head000.npz"))
    base = [z0["vert_lh"], z0["vert_rh"]]
    for h in range(h0, h1):
        f = os.path.join(OUT, f"head{h:03d}.npz")
        if os.path.exists(f):
            continue
        t = time.time()
        geom = H.make_geometry(h, P, base)
        ref = np.load(os.path.join(H.CACHE0, f"head{h:03d}.npz"))
        Gc = []
        for c in CANDS:
            G, _, pos = H.forward_for(geom, H.bem_for(geom, c), geom["E"])
            Gc.append(G)
        assert np.allclose(pos, ref["pos"], atol=1e-6), "다시 만든 머리가 저장된 머리와 다름"
        bem_t = H.bem_for(geom, geom["skull_true"])
        Z = np.random.default_rng(7000 + h).normal(size=geom["E"].shape)
        Gj = [H.forward_for(geom, bem_t, geom["E"] + Z * j / 1000)[0] for j in JITTERS]
        np.savez_compressed(f, G_cand=np.array(Gc, np.float32), cands=CANDS, G_jit=np.array(Gj, np.float32),
                            jitters=JITTERS, skull_true=geom["skull_true"], pos=pos)
        print(f"head{h:03d}: 진짜 전도도 {geom['skull_true']:.4f}, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
