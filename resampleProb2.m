addpath('/home/haol/matlab');

tmpdir='/tmp/parc';
testID = 1;
data=load('../parc/test/test.mat');
fname = strtrim(data.fname(testID,:));
reg = strtrim(data.reg(testID,:));

%
%fname = strsplit(fname,{'.','/'});
%fname = fname{9};
%

sphere_root = sprintf('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/%s/curves/',fname);
subj_sphere = sprintf('%s/lh.sphere.reflect.%s.vtk', sphere_root,reg);
ico_sphere = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci/icosphere_5.vtk';
prob = squeeze(data.prob(testID,:,:));
mkdir(tmpdir);
prob_list = {};
for i = 1: size(prob,1)
    pfile = sprintf('%s/parc.prob%d.txt',tmpdir,i);
    prob_list = [prob_list pfile];
    fp = fopen(pfile,'w');
    fprintf(fp, '%f\n', prob(i,:));
    fclose(fp);
end
plist = sprintf('%s,',prob_list{:});
plist = plist(1:end-1);
system(sprintf('/fs4/masi/haol/SurfRemesh -p %s -r %s -t %s --noheader --outputProperty %s/parc', plist,subj_sphere,ico_sphere,tmpdir));
%     [~,prob_]=max(squeeze(data.prob));
%     prob_=prob_-1;

prob_new = [];
for i = 1: size(prob,1)
    pfile = sprintf('%s/parc.prob%d.txt',tmpdir,i);
    prob_new = [prob_new; load(pfile)'];
end

[~,output]=max(prob_new);
output = output - 1;

truth = load(sprintf('%s/lh.label.txt',sphere_root));
[v,f] = read_vtk(sprintf('%s/lh.white.vtk',sphere_root));

system(['rm -rf ' tmpdir]);
write_property('../parc/test.vtk',v,f,struct('pred',output,'truth',truth));
disp('done');
