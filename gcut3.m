function parc = gcut3(v,f,prob)
    %% graph
    F=f+1;
    n = max(F(:));
    % remove duplicated edges
    if size(F,2) == 2
      rows = [F(:,1); F(:,2)];
      cols = [F(:,2); F(:,1)];
    else
      rows = [F(:,1); F(:,1); F(:,2); F(:,2); F(:,3); F(:,3)];
      cols = [F(:,2); F(:,3); F(:,1); F(:,3); F(:,1); F(:,2)];
    end
    rc = unique([rows,cols], 'rows','first');

    % fill adjacency matrix
    G = graph(sparse(rc(:,1),rc(:,2),1,n,n));

    %%
    % smoothness cost
    SmoothnessCost = ones(20) - eye(20);

    % data cost
    % softmax and -log
    % prob = (1 + exp(-prob)) .^ -1);
    prob = -log(prob);
    DataCost = prob;

    len = v(G.Edges.EndNodes(:,1),:)-v(G.Edges.EndNodes(:,2),:);
    len = sqrt(sum(len .* len,2));

    % sparse cost
    SparseSmoothness = sparse([G.Edges.EndNodes(:,1);G.Edges.EndNodes(:,2)],[G.Edges.EndNodes(:,2);G.Edges.EndNodes(:,1)],exp(-[len;len]),n,n);
    % SparseSmoothness = sparse([G.Edges.EndNodes(:,1);G.Edges.EndNodes(:,2)],[G.Edges.EndNodes(:,2);G.Edges.EndNodes(:,1)],1,n,n);

    [~,output]=min(prob);
    
    % graphcut
    [gch] = GraphCut('open', DataCost, SmoothnessCost, SparseSmoothness);
    [gch] = GraphCut('set', gch, output-1);
    [gch, L] = GraphCut('expand',gch);
    % [gch, L] = GraphCut('swap',gch);
    GraphCut('close', gch);
    
    % label
    parc = L';

    %% evaluation (this is not necessary)
%     disp(sum(parc ~= double(data.output(testID,:))))
%     disp([mean(similarity(data.output(testID,:)',data.target(testID,:)')), mean(similarity(data.target(testID,:)',parc'))])
end