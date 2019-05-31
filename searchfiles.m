% flie list and read vtks from /fs4/masi/lyui/CASCIO_CM? assignment

%[v,f]=read_vtk('/fs4/masi/lyui/CASCIO_CM/127462/127462/CASCIO_CM-x-127462-x-127462-x-surf_postproc_v1-x-d5a57784-4a10-4f42-a8e7-d9560915eba9/Surface_Reg/lh.mid.reg.vtk');
addpath('/home/haol/matlab');
folder = '/fs4/masi/lyui/CASCIO_CM';
pattern = fullfile(folder,'./*/*/*post*/Surface_Reg/lh.mid.reg.vtk');
listfiles = rdir(pattern);

X = [];
for i = 1:length(listfiles)
    filename = listfiles(i).name;
    [v, f] = read_vtk(filename);
    v1 = reshape(v,[],1);
    X = [X v1];
end

% Procrustes
S1 = reshape(X(:,1),[],3);
S2 = reshape(X(:,2),[],3);
[d, Z, tr] = procrustes(S1, S2);
write_vtk('/tmp/S1.vtk',S1,f);
write_vtk('/tmp/S2.vtk',S2,f);
write_vtk('/tmp/Z.vtk',Z,f);
Xhat = X(:,1);
for i=2:ncol
    S = reshape(X(:,i),[],3);
    [d1, Z1] = procrustes(S1, S);
    Shat = reshape(Z1,[],1);
    Xhat = [Xhat Shat];
end

% PCA 
[nrow, ncol]=size(Xhat);
mu = mean(Xhat,2);
M = repmat(mu,1,ncol);
A = Xhat-M;
[u,D] = eig(A'*A);
V=A*u;
eigvector = V./repmat(sqrt(sum(V.*V)),nrow,1);
diagonal = diag(D);
[~,I] = max(diagonal);
PC = eigvector(:,I-1:I);
PC1 = A'*PC(:,2);
PC2 = A'*PC(:,1);
scatter(PC1, PC2);

coeff = pca(Xhat');

% K means
K = 4;
Y1 = [PC1(:), PC2(:)];
Y = Y1';
[features, subj] = size(Y);
randIdx = randi(subj,[1 K]);
PC1 = A'*PC(:,2);
initCentroid = Y(:,randIdx);
cluster = zeros(1,subj);
oldCluster = cluster;
stop = false;

dist = [];
while stop == false
    for i = 1:subj
        for j = 1:K
            dist(j) = norm(Y(:,i)-initCentroid(:,j));
        end
        [~,indexClosest] = min(dist);
        cluster(i) = indexClosest;
    end
    recompCentr = zeros(features,K);
    for h = 1:K
        recompCentr(:,h) = mean(Y(:,cluster ==h),2);
    end
    oldCluster = cluster;
    if oldCluster == cluster
        stop = true;
    end
end

cmap = parula(K);
figure;
D = cmap(cluster,:);
hold on;
scatter (Y1(:,1),Y1(:,2),36,D);
hold on;
scatter(recompCentr(1,:), recompCentr(2,:),50,'red','+');
hold off;


% Find the mean shape and do PCA on first 20th point
avg = [];
coord = [];
PC1_val = [];
cnt = 0;
for j =1:ncol 
    cnt = cnt+1;
    % r is 163842x3 matrix, representing x y z of each vertex
    r = reshape(Xhat(:,j),[],3); 
    Vmean = mean(r,1);
    % each col of avg contains means of [x y z] for each vertex 
    avg = [avg Vmean']; 
    if cnt <= 20
        PC_ = pca(r);
        PC1_val = [PC1_val r*PC_(:, 1)];
        coord = [coord PC_]; 
    end
end
write_vtk('/tmp/mean.vtk', avg, f);








