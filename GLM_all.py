import os
import shutil
import h5py
import numpy as np
import pandas as pd
from numpy import unwrap
from scipy import sparse
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import PoissonRegressor
from joblib import Parallel, delayed
from tqdm import tqdm
# 1) 抬头余弦基底 & 快速卷积函数
def create_raised_cosine_basis(n_basis=16, history_ms=160, dt=1):
    ttb = np.arange(dt, history_ms+dt, dt, dtype=np.float32)
    log_t = np.log(ttb)
    centers = np.linspace(log_t[0], log_t[-1], n_basis, dtype=np.float32)
    width = centers[1] - centers[0]
    basis = [ ((np.cos(np.clip((log_t-c)*np.pi/(2*width), -np.pi, np.pi))+1)/2).astype(np.float32)
              for c in centers ]
    return np.stack(basis, axis=1)

def spike_history_design_fast(spikes, basis):
    T, K = spikes.shape[0], basis.shape[1]
    rev = basis[::-1,:]
    # 每个基底做一次卷积
    cols = [ np.convolve(spikes, rev[:,k], mode="full")[:T].astype(np.float32)
             for k in range(K) ]
    return np.stack(cols, axis=1)  # (T,K)

def fit_neuron_sparse(i, tr, va,
                      Xb_sparse, Xb_val_sparse,
                      y_tr, y_va,
                      spike_arr, history_basis):
    # 生成历史设计 (T,K)
    spikes = spike_arr[:, i].astype(np.float32)
    H = spike_history_design_fast(spikes, history_basis)
    H_tr, H_va = H[tr], H[va]
    # 稀疏拼接：行为特征已在 Xb_sparse 中；history 直接 csr
    Xtr = sparse.hstack([Xb_sparse, sparse.csr_matrix(H_tr)], format="csr")
    Xva = sparse.hstack([Xb_val_sparse, sparse.csr_matrix(H_va)], format="csr")
    # 拟合 Poisson GLM
    y_train_i = y_tr[:,i]
    y_val_i   = y_va[:,i]
    mdl = PoissonRegressor(alpha=0.001, max_iter=10000, fit_intercept=True)
    mdl.fit(Xtr, y_train_i)
    mu_tr = mdl.predict(Xtr)
    mu_va = mdl.predict(Xva)
    coef = mdl.coef_.astype(np.float32)
    intercept = np.float32(mdl.intercept_)
    # pseudo-R2
    def _dev(y, yp):
        eps=1e-8
        return 2*np.sum(y*np.log((y+eps)/(yp+eps)) - (y-yp))
    r2_tr = 1 - _dev(y_train_i,mu_tr)/_dev(y_train_i, y_train_i.mean())
    r2_va = 1 - _dev(y_val_i,mu_va)/_dev(y_val_i, y_val_i.mean())
    # Fisher 信息算 p‐value
    W = mu_tr

    reg = mdl.alpha  # 0.001
    Wmat = sparse.diags(W, format="csr")  # (n, n)
    F = Xtr.T @ (Wmat @ Xtr)
    F += reg * sparse.eye(Xtr.shape[1], format="csr")
    Cov = np.linalg.inv(F.toarray()).astype(np.float32)
    se  = np.sqrt(np.maximum(np.diag(Cov),1e-12)).astype(np.float32)
    z   = coef/(se+1e-8)
    pcoef = (2*(1-norm.cdf(np.abs(z)))).astype(np.float32)

    # 拼 intercept
    coefs = np.concatenate([coef, [intercept]]).astype(np.float32)
    pvals = np.concatenate([pcoef, [np.nan]]).astype(np.float32)
    return coefs, pvals, r2_tr, r2_va, mu_va.astype(np.float32)

# -----------------------------------------------------------------------------
# 2) 读取原始，并对齐到 1000Hz
# -----------------------------------------------------------------------------
# 2.1 Spike
with h5py.File("spike_binary_200Hz.h5","r") as hf:
    spike_arr = hf["spike_binary"][:].astype(np.float32)
neuron_names = [f"neuron_{i+1}" for i in range(spike_arr.shape[1])]
spike_df = pd.DataFrame(spike_arr, columns=neuron_names)

# 2.2 行为(原始) + 位置 + DLC + IMU
beh = pd.read_csv("F5D10_outdoor_modified.csv").drop(columns=["Index"],errors="ignore")
pos = pd.read_csv("positions_F5D10_outdoor.csv")[["head_x","head_y"]]
dlc = pd.read_csv("final_filtered_F5D10_outdoor_50hz.csv")[["head_v","bodyCenter1_v"]]
imu = pd.read_csv("F5D10_outdoor_IMU_features_basic.csv")[["roll","yaw","pitch","speed_z"]]

# 2.3 位置分箱
def bin_col(s,b, mn=None, mx=None):
    if mn is None: mn=s.min()
    if mx is None: mx=s.max()
    edges = np.linspace(mn,mx,b+1)
    out = np.digitize(s,edges)-1
    return np.clip(out,0,b-1).astype(np.int16)

pos["x_bin"] = (pos["head_x"]//(4*4.611473)).astype(int)
pos["y_bin"] = (pos["head_y"]//(4*4.611473)).astype(int)
uniq = pos[["x_bin","y_bin"]].drop_duplicates().reset_index(drop=True)
uniq["pos_idx"]=np.arange(len(uniq))
pos = pos.merge(uniq,on=["x_bin","y_bin"],how="left")

# 2.4 上采样到1000Hz
def repeat_df(df,f=4):
    return df.loc[df.index.repeat(f)].reset_index(drop=True)
def upsample_df(df,f=4):
    old=np.arange(len(df))
    new=np.linspace(0,len(df)-1,len(df)*f)
    return pd.DataFrame({c:np.interp(new,old,df[c].values).astype(np.float32)
                         for c in df.columns})

beh_2h = repeat_df(beh)
dlc_2h = upsample_df(dlc)
imu["roll"]=unwrap(imu["roll"])
imu["yaw"] =unwrap(imu["yaw"])
imu["pitch"]=unwrap(imu["pitch"])
imu_2h = upsample_df(imu)
pos_2h = repeat_df(pos[["pos_idx"]]).rename(columns={"pos_idx":"position"})

# 2.5 分箱 DLC/IMU
dlc_bin = pd.DataFrame({c: (bin_col(dlc_2h[c],24,-180,180) if "angle" in c else bin_col(dlc_2h[c],30))
                       for c in dlc_2h})
imu_bin = pd.DataFrame({c: (bin_col(imu_2h[c],24,-np.pi,np.pi) if c in ["roll","yaw","pitch"]
                            else bin_col(imu_2h[c],30))
                       for c in imu_2h})

# -----------------------------------------------------------------------------
# 3) OneHotEncoder（只对分箱列） + 保留所有箱
# -----------------------------------------------------------------------------
cat_df = pd.concat([dlc_bin, imu_bin, pos_2h],axis=1).reset_index(drop=True)

# 为每一列显式提供 categories
cats = []
for col in cat_df:
    if col in ["roll","yaw","pitch"]:
        cats.append(np.arange(24))
    elif col in ["head_v","bodyCenter1_v","speed_z"]:
        cats.append(np.arange(30))
    elif col=="position":
        cats.append(np.arange(uniq.shape[0]))
    else:
        cats.append(np.arange(30))
encoder = OneHotEncoder(categories=cats,
                        sparse_output=True,
                        handle_unknown="ignore")
X_cat = encoder.fit_transform(cat_df).astype(np.float32)  # CSR

# -----------------------------------------------------------------------------
# 4) 把行为特征(连续)、one-hot 特征拼成最终 CSR
# -----------------------------------------------------------------------------
X_beh = sparse.csr_matrix(beh_2h.values.astype(np.float32))  # (T, B)
X_both = sparse.hstack([X_beh, X_cat],format="csr")           # (T, B+P)

# -----------------------------------------------------------------------------
# 5) 准备 mapping 文件
# -----------------------------------------------------------------------------
feat_beh = list(beh_2h.columns)
feat_cat = encoder.get_feature_names_out(cat_df.columns.tolist())
history_basis = create_raised_cosine_basis(16,160)
hist_names = [f"history_{i+1}" for i in range(history_basis.shape[1])]
all_names  = feat_beh + list(feat_cat) + hist_names + ["intercept"]

os.makedirs("cv_results",exist_ok=True)
with open("cv_results/feature_mapping.txt","w") as f1, \
     open("cv_results/coefficients_mapping.txt","w") as f2:
    for idx,name in enumerate(all_names):
        f1.write(f"{idx}: {name}\n")
        f2.write(f"{idx}: {name}\n")
shutil.copy("cv_results/coefficients_mapping.txt",
            "cv_results/pvalue_mapping.txt")

# -----------------------------------------------------------------------------
# 6) 对齐 spike & 特征，CV 拆分并行拟合
# -----------------------------------------------------------------------------
start = int(1.238591123957225e4*200)
end   = int(1.588404693733597e4*200)
Y     = spike_df.iloc[start:end].reset_index(drop=True).values.astype(np.float32)
X_al  = X_both[:end-start]  # CSR

kf  = KFold(n_splits=5,shuffle=False)
spl = list(kf.split(X_al))

for fold in range(1,6):
    print(f"fold: {fold}")
    tr,va = spl[fold-1]
    Xtr_s = X_al[tr]; Xva_s = X_al[va]
    ytr   = Y[tr];    yva    = Y[va]

    results = Parallel(n_jobs=24, backend="threading")(
        delayed(fit_neuron_sparse)(
            i, tr, va,
            Xtr_s, Xva_s,
            ytr, yva,
            spike_df.iloc[start:end].values,
            history_basis
        )
        for i in tqdm(range(len(neuron_names)))
    )

    betas,pvals,r2t,r2v,pr = zip(*results)
    Bmat = np.stack(betas,axis=0)
    Pmat = np.stack(pvals,axis=0)
    dfR  = pd.DataFrame({
      "neuron": neuron_names,
      "pseudo_R2_train": r2t,
      "pseudo_R2_val":   r2v
    })
    pd.DataFrame(Bmat, index=neuron_names)\
      .to_csv(f"cv_results/fold{fold}_train_coefficients.csv")
    pd.DataFrame(Pmat, index=neuron_names)\
      .to_csv(f"cv_results/fold{fold}_train_pvalues.csv")
    dfR.to_csv(f"cv_results/fold{fold}_r2.csv",index=False)

    with h5py.File(f"cv_results/fold{fold}_pred.h5","w") as hf:
        hf.create_dataset("pred",data=np.stack(pr,axis=1),compression="gzip")
        hf.create_dataset("true",data=yva,           compression="gzip")

print("✅ Done.")
