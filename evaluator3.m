%% one-time load
tmpdir = '/tmp/parc';
addpath('/home/haol/matlab');
%addpath(genpath('~/Tools/matlab/SurfLibrary'));
root = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/test_mat';
root2 = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci';
load(sprintf('%s/sample_surface.mat',root));
disp('done');

%%
flist = dir('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/deform/input_fs/*fs.reg*label.txt');
% flist = dir('/home/lyui/mount/nfs/Sulcus/NORA_PFCSulci/deform/input_reflect/*reg15*label.txt');
flist = flist(1:2:end);
parc = zeros(163842, length(flist));

for i = 1: length(flist)
    parc(:,i) = load([flist(i).folder filesep flist(i).name]);
end
ico_sphere = sprintf('%s/icosphere_7.vtk',root2);
flist2 = dir(sprintf('%s/Reg0_Epoch*.mat',root));

id = split({flist.name},'.');
id = squeeze(id(:,:,1));
id = sort(id);

%%
load(sprintf('%s/summary.reg0.mat',root));
samples = 3;

mean_dice = zeros(100, 4);
for epoch = 1:100
    for i = 1: samples
        result = stats{epoch,i};
        result(isinf(result)) = NaN;
        mean_dice(epoch,:) = mean_dice(epoch,:) + nanmean(result);
    end
    mean_dice(epoch,:) = mean_dice(epoch,:) / samples;
end
[max_dice,max_dice_id] = max(mean_dice(:,1));
flist2(max_dice_id).name
[max_dice_id max_dice]
[min_dist,min_dist_id] = min(mean_dice(:,2));
flist2(min_dist_id).name
[min_dist_id min_dist]

%%
load(sprintf('%s/summary.reg0.mat',root));
% load(sprintf('%s/Baseline3_Reg0_Epoch83.mat',root)); % max dice
% load(sprintf('%s/Baseline3_Reg0_Epoch73.mat',root)); % min dist

epoch = 83; % max dice
% epoch = 64; % min dist
sampleID = 3;

sphere_root = fileparts(sample{sampleID,2});

test = strtrim(sample{sampleID,1});
leave_one_out = find(strcmp(id, test));
prob = histc(parc(:, setdiff(1:length(flist), leave_one_out)),0:19,2);
prob = prob / (size(parc,2)-1);
prob = prob';

mkdir(tmpdir);
prob_list = {};
for j = 1: size(prob,1)
    pfile = sprintf('%s/parc.prob%d.txt',tmpdir,j);
    prob_list = [prob_list pfile];
    fp = fopen(pfile,'w');
    fprintf(fp, '%f\n', prob(j,:));
    fclose(fp);
end
plist = sprintf('%s,',prob_list{:});
plist = plist(1:end-1);
% sphere_subj = strrep(sample{sampleID,2},'reg0','reg15');
sphere_subj = strrep(sample{sampleID,2},'lh.sphere.reflect.reg0.vtk','lh.sphere.fs.reg.vtk');
system(sprintf('SurfRemesh -p %s -r %s -t %s --noheader --outputProperty %s/parc',plist,sphere_subj,ico_sphere,tmpdir));

prob_new = [];
for j = 1: size(prob,1)
    pfile = sprintf('%s/parc.prob%d.txt',tmpdir,j);
    prob_new = [prob_new; load(pfile)'];
end
system(['rm -rf ' tmpdir]);

pred_ma = gcut3(sample{sampleID,3},sample{sampleID,4},prob_new);

write_property('../parc/test.vtk', sample{sampleID,3}, sample{sampleID,4}, struct('truth', sample{sampleID,6}, 'pred', pred{epoch,sampleID}, 'ma', pred_ma));

disp('done');
