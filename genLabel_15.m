%% one-time load
addpath('/home/haol/matlab');
root = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/test_mat/dice';

fold = 1;
sampleID = 3;

load(sprintf('%s/sample_surface_CV%d.mat',root,fold));
disp('done.');

%%
load(sprintf('%s/summary.ma.mat',root));
ma = pred{fold, sampleID};
load(sprintf('%s/summary.Baseline.mat',root));
baseline = pred{fold, sampleID};
load(sprintf('%s/summary.CV.mat',root));
ours = pred{fold, sampleID};

% write_property('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/visual2.vtk', sample{sampleID,3}, sample{sampleID,4}, struct('truth', sample{sampleID,6}, 'ma',ma,'base',baseline,'ours',ours ));

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
truth=double(sample{sampleID,6});
ma(ma == 0) = curv(ma == 0);
baseline(baseline == 0) = curv(baseline == 0);
ours(ours == 0) = curv(ours == 0);
truth(truth == 0) = curv(truth == 0);
write_property('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/inflated.vtk', v, f, struct('truth', truth, 'ma',ma,'base',baseline,'ours',ours ));



disp('done.');


%% sample code
[v,f]=read_vtk('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/lh.white.mean.inflated.vtk');
write_property('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/new.vtk', v, f, struct('curve',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/lh.reflect.curv.txt'), 'iH', load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/lh.reflect.iH.txt'),'sulc',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/lh.reflect.sulc.txt')));

write_property('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/new1.vtk', v, f, struct('curve',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg0.curv.txt'), 'iH', load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg0.iH.txt'),'sulc',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg0.sulc.txt')));
write_property('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/new2.vtk', v, f, struct('curve',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg5.curv.txt'), 'iH', load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg5.iH.txt'),'sulc',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg5.sulc.txt')));
write_property('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/new3.vtk', v, f, struct('curve',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg10.curv.txt'), 'iH', load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg10.iH.txt'),'sulc',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg10.sulc.txt')));
write_property('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/new4.vtk', v, f, struct('curve',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg15.curv.txt'), 'iH', load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg15.iH.txt'),'sulc',load('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_reflect/n002t2.lh.reg15.sulc.txt')));
















