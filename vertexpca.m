clear 
addpath('/home/haol/matlab');
folder = '/fs4/masi/lyui/CASCIO_CM';
pattern = fullfile(folder,'./*/*/*post*/Surface_Reg/lh.mid.reg.vtk');
listfiles = rdir(pattern);

n_sample = 50;
listfiles = listfiles(1:n_sample);

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


% Vertexwise pca
PC1val = zeros(3, nrow/3);
projection = zeros(n_sample, nrow/3);
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
%     mu=mean(r, 2);
%     r1 = r-mu;
%     proj = r1'*PC_1;
    projection(:,j) = score(:,1);
end

% Find the mean shape 
Vm = mean(Xhat, 2);
Vmean = reshape(Vm, [], 3);
Vs = std(Xhat);


projmean = mean(abs(projection)); 
cmap1 = struct('cmap', projmean);
write_property('/tmp/meancolor.vtk',Vmean,f,cmap1);







