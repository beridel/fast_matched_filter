% :copyright:
%     William B. Frank and Eric Beauce
% :license:
%     GNU General Public License, Version 3
%     (https://www.gnu.org/licenses/gpl-3.0.en.html)

function [cc_sum] = fast_matched_filter(templates, moveouts, weights, data, step)
% input:
% templates ---------- 4D matrix [time x components x stations x templates]
% moveouts ----------- 3D matrix [components x stations x templates]
%                   or 2D matrix [stations x templates] 
%                   (in that case, the same moveout is attributed to each component)
% weights ------------ 3D matrix [components x stations x templates]
%                   or 2D matrix [stations x templates] 
%                   (in that case, the same weight is attributed to each component)
% data --------------- 3D matrix [time x components x stations]
% step --------------- interval between correlations (in samples)
%
% NB: Mean and trend MUST be removed from template and data traces before
%     using this function
%
% output:
% 2D matrix [times x templates (at step defined interval)]
n_samples_template = size(templates, 1);
n_components = size(templates, 2);
n_stations = size(templates, 3);
n_templates = size(templates, 4);
n_samples_data = size(data, 1);

sum_square_templates = squeeze(sum(templates .^ 2, 1));

%data_double_sq = double(data) .^ 2;
%csum_square_data = csum(data_double_sq, ...
%                        n_samples_template, ...
%                        n_samples_data, ...
%                        n_stations, ...
%                        n_components);
%csum_square_data = single(csum_square_data);
%clear data_double_sq;

n_corr = floor((n_samples_data - n_samples_template - max(moveouts(:))) / step);

% extend the moveouts and weights matrices from 2D to 3D matrices, if necessary
b = ones(n_components, 1);
if numel(moveouts) ~= n_components * n_stations * n_templates
    moveouts = reshape(kron(moveouts, b), [n_components, n_stations, n_templates]);
end
if numel(weights) ~= n_components * n_stations * n_templates
    weights = reshape(kron(weights, b), [n_components, n_stations, n_templates]);
end

% input arguments (brackets indicate a non-scalar variable):
% templates (float) [time x components x stations x templates]
% sum of square of templates (float) [components x stations x templates]
% moveouts (int) [components x stations x templates]
% data (float) [time x components x stations]
% weights (float) [components x stations x templates]
% step, or the samples between each sliding window (int)
% N samples per template trace (int)
% N samples per data trace (int)
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
                        weights, ...
                        step, ...
                        n_samples_template, ...
                        n_samples_data, ...
                        n_templates, ...
                        n_stations, ...
                        n_components, ...
                        n_corr);
                    
cc_sum = double(reshape(cc_sum, [], n_templates));
Nzeros = sum(cc_sum(1,:) == 0.);
if Nzeros > 10
    text = sprintf('%i correlation computations were skipped. Can be caused by zeros in data, or too low amplitudes (try to increase the gain).', Nzeros);
    disp(text)
end
end

