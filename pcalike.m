clear 
addpath('/home/haol/matlab');
% Read the files and create table
opts = detectImportOptions('/nfs/masi/lyui/CASCIO_CM/demo1.csv');
opts = setvartype(opts, {'xnat_session_number','dx_group','age_at_scan', 'gender'}, 'single');
T1 = readtable('/nfs/masi/lyui/CASCIO_CM/demo1.csv',opts);

opts2 = detectImportOptions('/nfs/masi/lyui/CASCIO_CM/demo2.csv');
opts2 = setvartype(opts2, {'age_at_dx'}, 'single');
T2 = readtable('/nfs/masi/lyui/CASCIO_CM/demo2.csv',opts2);

opts3 = detectImportOptions('/nfs/masi/lyui/CASCIO_CM/qa.csv');
opts3 = setvartype(opts3, {'quality'}, 'string');
T3 = readtable('/nfs/masi/lyui/CASCIO_CM/qa.csv', opts3);
T3.Properties.VariableNames{'session_label'} = 'xnat_session_number'; 

C = innerjoin(T1, T2);
newtable = C(:, {'xnat_session_number','dx_group','age_at_scan','age_at_dx', 'gender'});
T = rmmissing(newtable);

% Find the listfiles
folder = '/nfs/masi/lyui/CASCIO_CM';
pattern = fullfile(folder,'./*/*/*post*/Surface_Reg/lh.mid.reg.vtk');
listfiles = rdir(pattern);
valid_file_id = [];
for i = 1:length(listfiles)
    [~,name]=fileparts(fileparts(listfiles(i).folder));
    index = find(strcmp(T3.as_label, name));
    if strcmp(T3.quality(index), "Good")==1
        valid_file_id = [valid_file_id i];
    end
end
listfiles = listfiles(valid_file_id);

reference = zeros(length(listfiles),1); 
key = split(sprintf('%d ',T.xnat_session_number), ' ');
key(end) = [];
for i = 1:length(listfiles)
    filename = split(listfiles(i).name, '-x-');
    filename = filename{3};
    id = find(ismember(key, filename));
    id1 = find(ismember(str2num(filename), T.xnat_session_number));
    if ~isempty(id)
       reference(i) = id(1);
    end
end

% big step: remove duplicate!!! file contains unique subject names now
[c, ia, ic] = unique(reference);
new_ref = zeros(length(listfiles),1);
new_ref(ia(c>0)) = reference(ia(c>0));
file = {listfiles(new_ref==0).name};

X = [];
for i = 1:length(file)
    filename = file{i};
    v = read_vtk(filename);
    v1 = reshape(v,[],1);
    X = [X v1];
end
[~, f] = read_vtk(filename);

% Procrustes
[nrow, ncol]=size(X);
S1 = read_vtk('lh.avg.vtk');
Xhat = [];
for i=1:ncol
    S = reshape(X(:,i),[],3);
    [d1, Z1] = procrustes(S1, S, 'scaling', false, 'reflection', false);
    Shat = reshape(Z1,[],1);
    Xhat = [Xhat Shat];
end

% Vertexwise pca
PC1val = zeros(3, nrow/3);
projection = zeros(length(file), nrow/3);
for j =1:nrow/3 
    if mod(j, 10000)==0
        disp(j);
    end
    x = Xhat(j, :);
    y = Xhat(j+nrow/3,:);
    z = Xhat(j+nrow/3*2, :);
    r = [x;y;z];
    [PC_,score,~] = pca(r');
    
    % PC_1 is the first principle component
    PC_1 = PC_(:,1);
    PC1val(:,j) = PC_1;
    
    % Projection on PC_1
    projection(:,j) = score(:,1);
end


% PCA-like analysis on absolute projection (no mean subracted)
[row, col]=size(projection');
M = abs(projection');
[u,D] = eig(M'*M);
V=M*u;
eigvector = V./repmat(sqrt(sum(V.*V)),row,1);
diagonal = diag(D);
[~,I] = max(diagonal);
PC = eigvector(:,I-1:I);
PC1 = M'*PC(:,2);
PC2 = M'*PC(:,1);
scatter(PC1, PC2);
% Visualize 
Vm = mean(Xhat, 2);
Vmean = reshape(Vm, [], 3);
eig1 = PC(:,2);
eig2 = PC(:,1);
cmap = struct('cmap', eig1', 'cmap2', eig2');
write_property('/tmp/meanproj.vtk',Vmean,f,cmap);

%% Color the age info on PC1 and PC2
idx = find(new_ref~=0);
ageAtScan = table2array(T(new_ref(idx), 3));
ageAtDx = table2array(T(new_ref(idx), 4));

subplot(1,3,1);
ageAtDx2 = ageAtDx;
ageAtDx2(ageAtDx > 30) = 30;
scatter (PC1, PC2,72,ageAtDx2,'filled');
set(gca,'xticklabels',[],'yticklabels',[],'box','on');
xlabel('PC1'); ylabel('PC2'); title('age at dx');
colorbar('southoutside');
axis equal tight;

subplot(1,3,2);
scatter (PC1, PC2,72,(ageAtDx > 7),'filled');
set(gca,'xticklabels',[],'yticklabels',[],'box','on');
xlabel('PC1'); ylabel('PC2'); title('age at dx before/after 7');
colorbar('southoutside');
axis equal tight;

subplot(1,3,3);
ageAtScan2 = ageAtScan;
ageAtScan2(ageAtScan2 > 30) = 30;
scatter (PC1, PC2,72,ageAtScan2,'filled');
set(gca,'xticklabels',[],'yticklabels',[],'box','on');
xlabel('PC1'); ylabel('PC2'); title('age at scan');
colorbar('southoutside');
axis equal tight;

colormap jet;

%%
% Find the outliers in the scatter plot
[~,I1] = sort(PC1); % Right*
[~,I2] = min(PC1); % Left
[~,I3] = sort(PC2); % Down*
[~,I4] = max(PC2); % Up
newT = T(new_ref(idx), :);
a = table2array(newT(I1(end-2), 3)); % CASIO 214882 CASIO_CM 215605
b = table2array(newT(I2, 3)); % CASIO 212467 CASIO_CM 233853
c = table2array(newT(I3(3), 3)); % CASIO 212519 CASIO_CM 227063
d = table2array(newT(I4, 3)); % CASIO_CM 226959

write_vtk('/tmp/a.vtk',reshape(Xhat(:,I1(end-2)),[],3),f);
write_vtk('/tmp/b.vtk',reshape(Xhat(:,I2),[],3),f);
write_vtk('/tmp/c.vtk',reshape(Xhat(:,I3(3)),[],3),f);
write_vtk('/tmp/d.vtk',reshape(Xhat(:,I4),[],3),f);

ii = find(table2array(newT(:,5))==1);
jj = find(table2array(newT(:,5))==0);
mean1 = mean(table2array(newT(ii,4)));
mean0 = mean(table2array(newT(jj,4)));
std1 = std(table2array(newT(ii,4)));
std0 = std(table2array(newT(jj,4)));
