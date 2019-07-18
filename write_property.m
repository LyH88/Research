function write_property(fn,v,f,property)
    fp = fopen(fn,'w');
    fprintf(fp, '# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n');
    fprintf(fp, 'POINTS %d float\n', size(v, 1));
    fprintf(fp, '%f %f %f\n', v');
    fprintf(fp, 'POLYGONS %d %d\n', size(f, 1), size(f, 1) * 4);
    fprintf(fp, '3 %d %d %d\n', f');
    fprintf(fp, 'POINT_DATA %d\n', size(v, 1));
    fprintf(fp, 'FIELD ScalarData %d\n', numel(fieldnames(property)));
    names = fieldnames(property);
    for i = 1: numel(fieldnames(property))
        fprintf(fp, '%s 1 %d float\n', names{i}, size(v, 1));
        fprintf(fp, '%f\n', getfield(property,names{i}));
    end
    fclose(fp);
end