tmpdir = '/tmp/parc';

root = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/naive_baseline1/';
root2 = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci';
surf_remesh = '/fs4/masi/haol/SurfRemesh';
flist = dir(sprintf('%s/Reg0_Epoch*.mat',root));
ico_sphere = sprintf('%s/icosphere_5.vtk',root2);
data = load([root filesep flist(1).name]);
disp('done');

%% preload
sample = cell(size(data.fname, 1),6);
for testID = 1: size(data.fname,1)
    sample(testID,1)={data.fname(testID,:)};
    fname = strtrim(data.fname(testID,:));
    reg = strtrim(data.reg(testID,:));
    hemi = 'lh';
    if contains(fname,'_r')
        fname = strsplit(fname,'_r');
        fname = fname{1};
        hemi = 'rh';
    end
    fprintf('%s.%s.%s\n',fname,hemi,reg);
    sphere_root = sprintf('%s/%s/curves/',root2,fname);
    subj_sphere = sprintf('%s/%s.sphere.reflect.%s.vtk', sphere_root,hemi,reg);
    [v,f] = read_vtk(sprintf('%s/%s.white.vtk',sphere_root,hemi));
    sample(testID,2)={subj_sphere};
    sample(testID,3)={v};
    sample(testID,4)={f};
    f = f + 1;
    nnv = cell(size(v, 1), 1);
    for i = 1: size(f, 1)
        nnv{f(i,1)} = [nnv{f(i,1)}, f(i, [2 3])];
        nnv{f(i,2)} = [nnv{f(i,2)}, f(i, [1 3])];
        nnv{f(i,3)} = [nnv{f(i,3)}, f(i, [1 2])];
    end
    for i = 1: size(v, 1)
        nnv{i} = unique(nnv{i});
    end
    sample(testID,5)={nnv};
    sample(testID,6)={load(sprintf('%s/%s.label.txt', sphere_root,hemi))};
end
save(sprintf('%s/sample_surface.mat',root),'sample');
disp('done');

%% evaluation
skip_id = 0;
load(sprintf('%s/sample_surface.mat',root));

stats = cell(length(flist),size(data.fname, 1));
pred = cell(length(flist),size(data.fname, 1));
if exist(sprintf('%s/summary.reg0.mat',root),'file')
    load(sprintf('%s/summary.reg0.mat',root));
    begin_id = find(cellfun(@isempty, stats(:,1)),1);
    if (begin_id <= skip_id)
        begin_id = skip_id + 1;
    end
else
    begin_id = skip_id + 1;
end

for i = begin_id: length(flist)
    fprintf('Epoch: %d\n', i);
    data = load([root filesep flist(i).name]);
    for k = 1: size(sample,1)
        fprintf(' Sample: %d\n', k);
        testID = find(strcmp(cellstr(data.fname),strtrim(sample{k})));
        prob = squeeze(data.prob(testID,:,:));
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
        system(sprintf('%s -p %s -r %s -t %s --noheader --outputProperty %s/parc',surf_remesh,plist,sample{k,2},ico_sphere,tmpdir));

        prob_new = [];
        for j = 1: size(prob,1)
            pfile = sprintf('%s/parc.prob%d.txt',tmpdir,j);
            prob_new = [prob_new; load(pfile)'];
        end
        system(['rm -rf ' tmpdir]);

%         [~,output] = max(prob_new);
%         output = output - 1;
        output = gcut(sample{k,3},sample{k,4},prob_new);

        truth = sample{k,6};

%         write_property('../parc/test.vtk',sample{k,3},sample{k,4},struct('pred',output,'truth',truth));
        result=similarity(output',truth);
%         mean(result)
        result = [result dist(output',truth,sample{k,3},sample{k,5})];
        stats(i,k) = {result};
        pred(i,k)={output};
        disp('done');
    end
    save(sprintf('%s/summary.reg0.mat',root),'stats','pred');
end
