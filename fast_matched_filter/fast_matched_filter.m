function [cc_sum] = fast_matched_filter(templates, moveouts, weights, data, step)
n_samples_template = size(templates, 1);
n_components = size(templates, 2);
n_stations = size(templates, 3);
n_templates = size(templates, 4);
n_samples_data = size(data, 1);

sum_square_templates = zeros(n_components, n_stations, n_templates);
for t = 1:n_templates
    for s = 1:n_stations
        for c = 1:n_components
            sum_square_templates(c,s,t) = sum(templates(:,c,s,t) .^ 2);
        end
    end
end

csum_square_data = zeros(n_samples_data + 1, n_components, n_stations);
csum_square_data(2:end,:,:) = cumsum(data .^ 2);

n_corr = floor((n_samples_data - n_samples_template - max(moveouts(:))) / step);

% input arguments (brackets indicate a non-scalar variable):
% templates (float) [time x components x stations x templates]
% sum of square of templates (float) [components x stations x templates]
% N samples of templates (int) [stations x templates]
% moveouts (int) [stations x templates]
% data (float) [time x components x stations]
% data squared (float) [time x components x stations]
% N samples per data trace (int)
% step, or the samples between each sliding window (int)
% N templates (int)
% N stations (int)
% N components (int)
% N samples in correlation sums (int)

% output arguments:
% cc_sum (float)

templates = single(templates(:));
sum_square_templates = single(sum_square_templates(:));
moveouts = int32(moveouts(:));
data = single(data(:));
csum_square_data = single(csum_square_data(:));
weights = single(weights(:));
step = int32(step);
n_samples_template = int32(n_samples_template);
n_samples_data = int32(n_samples_data);
n_templates = int32(n_templates);
n_stations = int32(n_stations);
n_components = int32(n_components);
n_corr = int32(n_corr);

cc_sum = matched_filter(templates, ...
                        sum_square_templates, ...
                        moveouts, ...
                        data, ...
                        csum_square_data, ...
                        weights, ...
                        step, ...
                        n_samples_template, ...
                        n_samples_data, ...
                        n_templates, ...
                        n_stations, ...
                        n_components, ...
                        n_corr);
                    
cc_sum = double(reshape(cc_sum, [], n_templates));
end

