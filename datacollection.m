addpath('/home/haol/matlab');

% Read the files and create table
opts = detectImportOptions('/fs4/masi/lyui/CASCIO_CM/demo1.csv');
opts = setvartype(opts, {'xnat_session_number','dx_group','age_at_scan'}, 'single');
T1 = readtable('/fs4/masi/lyui/CASCIO_CM/demo1.csv',opts);

opts2 = detectImportOptions('/fs4/masi/lyui/CASCIO_CM/demo2.csv');
opts2 = setvartype(opts2, {'age_at_dx'}, 'single');
T2 = readtable('/fs4/masi/lyui/CASCIO_CM/demo2.csv',opts2);

C = innerjoin(T1, T2);
table = C(:, {'xnat_session_number','dx_group','age_at_scan','age_at_dx'});
T = rmmissing(table);

% Find the listfiles
folder = '/fs4/masi/lyui/CASCIO_CM';
pattern = fullfile(folder,'./*/*/*post*/Surface_Reg/lh.mid.reg.vtk');
listfiles = rdir(pattern);


reference = zeros(length(listfiles),1);
for i = 1:length(listfiles)
    filename = split(listfiles(i).name, '-x-');
    filename = filename{3};
    key = split(sprintf('%d ',T.xnat_session_number), ' ');
    key(end) = [];
    id = find(ismember(key, filename));
    if ~isempty(id)
        reference(i) = id;
    end
end

% big step: remove duplicate!!!
[c, ia, ic] = unique(reference);
new_ref = zeros(length(listfiles),1);
new_ref(ia) = reference(ia);

file = {listfiles(new_ref~=0).name};



