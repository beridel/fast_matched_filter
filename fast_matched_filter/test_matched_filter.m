% :copyright:
%     William B. Frank and Eric Beauce
% :license:
%     GNU General Public License, Version 3
%     (https://www.gnu.org/licenses/gpl-3.0.en.html)

%NB all data and templates must have their mean removed
% otherwise, simplified correlation coefficient equation is no
% longer valid!!!
% this is already the case for this mean zero random data/templates
% but not necessarily the case for real data

%% define network and waveforms
sampling_rate = 100;
n_templates = 1;
n_stations = 1;
n_components = 5;
template_duration = 10;
data_duration = 86400;
step = 1;

%% define templates
template_start_times = round(rand(1, n_templates) * data_duration / 2) + 1; % determines the time (in seconds) within the data to extract a template
moveouts = zeros(n_stations, n_templates); % in seconds

min_moveout = 0
max_moveout = 10
moveouts = zeros(n_stations, n_templates)
for t = 1:n_templates
    moveouts(:,t) = rand(n_stations, 1) * (max_moveout - min_moveout) + min_moveout
end
moveouts = round(moveouts * sampling_rate)

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
cc_sum = run_matched_filter(templates, moveouts, weights, data, step);
fprintf('Done step %d in %.2f seconds!\n', toc);

