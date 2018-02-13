% :copyright:
%     William B. Frank and Eric Beauce
% :license:
%     GNU General Public License, Version 3
%     (https://www.gnu.org/licenses/gpl-3.0.en.html)

%NB all data must be detrended / highpass filtered and templates 
% must have their mean removed otherwise, 
% simplified correlation coefficient equation is no
% longer valid!!! (cf. consequences_nonzero.pdf)
% this is already the case for this mean zero random data/templates
% but not necessarily the case for real data


%% define network and waveforms
sampling_rate = 10;
n_templates = 2;
n_stations = 1;
n_components = 1;
template_duration = 10;
data_duration = 86400;
step = 1;

% determines the time (in seconds) within the data to extract a template
template_start_times = round(rand(1, n_templates) * data_duration / 2) + 1;

min_moveout = 0;
max_moveout = 10;
moveouts = zeros(n_stations, n_templates);
for t = 1:n_templates
    moveouts(:,t) = rand(n_stations, 1) * (max_moveout - min_moveout) + min_moveout;
end
moveouts = round(moveouts * sampling_rate);

%% generate random data
n_samples_data = data_duration * sampling_rate;
data = rand(n_samples_data, n_components, n_stations);
for s = 1:n_stations
    for c = 1:n_components
        data(:,c,s) = data(:,c,s) - mean(data(:,c,s));
    end
end

%% generate templates from data by extracting the waveforms at template_times
n_samples_templates = template_duration * sampling_rate;
moveouts = moveouts * sampling_rate;

templates = zeros(max(n_samples_templates(:)), n_components, n_stations, n_templates);
for t = 1:n_templates
    start_t = template_start_times(t) * sampling_rate + 1;
    
    for s = 1:n_stations
        stop_t = start_t + template_duration * sampling_rate - 1;

        % adjust for station moveout
        start = start_t + round(moveouts(s,t));
        stop = stop_t + round(moveouts(s,t));
        
        template = data(start:stop,:,s);
        templates(:,:,s,t) = bsxfun(@minus, template, mean(template, 1));
    end
end

weights = ones(n_stations, n_templates) / n_stations;

%% C matched filter
tic;
cc_sum = fast_matched_filter(templates, moveouts, weights, data, step);
fprintf('Done in %.2f seconds!\n', toc);

%% Check the accuracy 
for t =1:n_templates
    fprintf('========================================\n');
    fprintf('Template %i was extracted from synthetic data at time %.1fsec.\n', t, template_start_times(1,t));
    max_cc = max(cc_sum(:,t));
    time_max_cc = find(cc_sum(:,t) == max_cc) / sampling_rate;
    fprintf('Maximum correlation (%.2f) found at time %.2fsec.\n', max_cc, time_max_cc);
end

