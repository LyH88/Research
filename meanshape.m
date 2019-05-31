addpath('/home/haol/matlab');
folder = '/fs4/masi/lyui/CASCIO_CM';
pattern = fullfile(folder,'./*/*/*post*/Surface_Reg/lh.mid.reg.vtk');
listfiles = rdir(pattern);

listfiles = listfiles(1:10);

X = [];
for i = 1:length(listfiles)
    disp(i);
    filename = listfiles(i).name;
    tic;
    v = read_vtk(filename);
    toc;
    v1 = reshape(v,[],1);
    X = [X v1];
end
[~, f] = read_vtk(filename);

% Procrustes
[nrow, ncol]=size(X);
S1 = reshape(X(:,1),[],3);
Xhat = X(:,1);
for i=2:ncol
    S = reshape(X(:,i),[],3);
    [d1, Z1] = procrustes(S1, S, 'scaling', false, 'reflection', false);
    Shat = reshape(Z1,[],1);
    Xhat = [Xhat Shat];
end

% Find the mean shape 
Vm = mean(Xhat, 2);
Vmean = reshape(Vm, [], 3);
Vs = std(Xhat);
id = 1: 163842;
id2 = -163842:-1;
cmap = struct('cmap', id, 'cmap2', id2);
write_property('/tmp/mean2.vtk',Vmean,f,cmap);
write_vtk('/tmp/mean.vtk', Vmean, f);






