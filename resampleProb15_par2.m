function resampleProb15_par2(i)
    tmpdir = sprintf('/tmp/parc%d',i);

    root = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/test_mat/dice';
    root2 = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci';
    output_dir = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/test_mat';
    surf_remesh = '/fs4/masi/haol/SurfRemesh';
    flist = '/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/test_mat/dice/Baseline3_Reg0_Epoch83.mat';
    ico_sphere = sprintf('%s/icosphere_5.vtk',root2);
    data = load(flist);

    %% evaluation
    load(sprintf('%s/sample_surface.mat',root));

    stats = cell(1,size(data.fname, 1));
    pred = cell(1,size(data.fname, 1));
    
    if ~exist(sprintf('%s/summary.reg0.%d.mat',output_dir,i),'file')
        fprintf('Epoch: %d\n', i);
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
            stats(1,k) = {result};
            pred(1,k)={output};
            disp('done');
        end
    end
    [~,tar] = fileparts(root);
    output_dir = sprintf('%s/%s',output_dir,tar);
    system(sprintf('mkdir -p %s',output_dir));
    save(sprintf('%s/summary.reg0.%0#3d.mat',output_dir,i),'stats','pred');
    exit(0);
end
