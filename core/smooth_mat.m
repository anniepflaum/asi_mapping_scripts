core_dir = fileparts(mfilename('fullpath'));
scripts_dir = fileparts(core_dir);
project_dir = fileparts(scripts_dir);
receiver_data_dir = fullfile(project_dir, 'receiver_data');

smooth_span = 9750;
smooth_method = 'moving';

mat_files = dir(fullfile(receiver_data_dir, '*', '*.mat'));

if isempty(mat_files)
    error('No receiver MAT files found under: %s', receiver_data_dir);
end

processed_count = 0;
skipped_count = 0;

for k = 1:numel(mat_files)
    mat_path = fullfile(mat_files(k).folder, mat_files(k).name);
    [folder, base_name, ~] = fileparts(mat_path);
    lower_base_name = lower(base_name);

    if endsWith(lower_base_name, '_smooth') || endsWith(lower_base_name, '_minus_smooth')
        skipped_count = skipped_count + 1;
        continue;
    end

    output_path = fullfile(folder, [base_name '_minus_smooth.mat']);
    data = load(mat_path);

    if ~isfield(data, 'flighttime') || ~isfield(data, 'faradayangle')
        warning('Skipping %s: missing flighttime or faradayangle.', mat_path);
        skipped_count = skipped_count + 1;
        continue;
    end

    flighttime = data.flighttime(:);
    faradayangle = abs(data.faradayangle(:));

    if isempty(faradayangle)
        warning('Skipping %s: faradayangle is empty.', mat_path);
        skipped_count = skipped_count + 1;
        continue;
    end

    file_smooth_span = min(smooth_span, numel(faradayangle));
    smoothed_faraday = smooth(faradayangle, file_smooth_span, smooth_method);
    faraday_minus_smoothed = faradayangle - smoothed_faraday;
    source_mat_path = mat_path;

    save( ...
        output_path, ...
        'flighttime', ...
        'faradayangle', ...
        'faraday_minus_smoothed', ...
        'smoothed_faraday', ...
        'smooth_span', ...
        'file_smooth_span', ...
        'smooth_method', ...
        'source_mat_path' ...
    );

    processed_count = processed_count + 1;
    disp(['Saved raw-minus-smoothed data to: ' output_path]);
end

fprintf('Processed %d receiver MAT files; skipped %d files.\n', processed_count, skipped_count);
