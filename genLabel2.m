%% one-time load
addpath(genpath('~/Tools/matlab/SurfLibrary'));
root = '/home/lyui/mount/nfs/Sulcus/NORA_PFCSulci/parc/test_mat/dice';

fold = 1;
sampleID = 1;

load(sprintf('%s/sample_surface_CV%d.mat',root,fold));

%%
load(sprintf('%s/summary.ma.mat',root));
ma = pred{fold, sampleID};
load(sprintf('%s/summary.Baseline.mat',root));
baseline = pred{fold, sampleID};
load(sprintf('%s/summary.CV.mat',root));
ours = pred{fold, sampleID};

% write_property('/home/lyui/mount/nfs/Sulcus/NORA_PFCSulci/parc/test.vtk', sample{sampleID,3}, sample{sampleID,4}, struct('truth', sample{sampleID,6}, 'ma',ma,'base',baseline,'ours',ours ));

% For inflated surface
[v,f]=read_vtk(strrep(strrep(sample{sampleID,2},'sphere.reflect.reg0','inflated'),'/home/lyui/mount/nfs/','/nfs/masi/lyui/'));
curv = load(strrep(strrep(sample{sampleID,2},'sphere.reflect.reg0.vtk','curv.txt'),'/home/lyui/mount/nfs/','/nfs/masi/lyui/'));
curv = -curv;
curv(curv > 1) =1;
curv(curv < -1) = -1;
curv = (curv + 1)/4;
ma = double(ma);
baseline = double(baseline);
ours = double(ours);
ma(ma == 0) = curv(ma == 0);
baseline(baseline == 0) = curv(baseline == 0);
ours(ours == 0) = curv(ours == 0);
write_property('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/test.vtk', v, f, struct('truth', sample{sampleID,6}, 'ma',ma,'base',baseline,'ours',ours ));
