function [data, y, u] = suuvid_get_data(data_file, vo)
data = readtable(data_file,'Delimiter',',','ReadVariableNames',true, 'TreatAsEmpty',{'.','NA'});

n_t = size(data,1); %number of rows
observations_to_fit = 1:n_t; %fit all observations by default

%this will choke if the subject (or simulation) only has 1 unique button, but the model expects 2+
%n_actions = max(data{:, 'key'});
%y = zeros(n_actions+1, n_t); %last element is no response

y = zeros(vo.n_outputs, n_t); %last element is no response
for i = 1:n_t
    this_key = data{i, 'key'};
    if this_key > 0
        y(this_key, i) = 1; %populate element of y vector for chosen action
    else
        y(end, i) = 1; %populate no response
    end 
    
end

%right shift reinforcement
new_trial = [1; diff(data{observations_to_fit, 'instrial'})];
choice = data{observations_to_fit, 'key'};
reinforcement = data{observations_to_fit, 'nreward'};

%remove the curkey clairvoyance for now: generates a ramping up of predicted probability for first chosen action in a trial
zero_curkey=1;
for i = 1:length(new_trial)

    if choice(i) > 0
        zero_curkey = 0; %disable zeroing now that a key has been pressed
    elseif new_trial(i) == 1
        zero_curkey = 1; %on a new trial, enable zeroing out
        if i > 1, data{i-1, 'curkey'} = 0; end %also zero out preceding timestep because we double-lag curkey. This will ensure a 0 on a new trial
    end
    
    if zero_curkey == 1
        data{i, 'curkey'} = 0;
    end
    
end



%right shift everything by 1 to get learning rule/evolution right
u = [ [ 0; new_trial(1:end-1) ], ...
    [ 0; choice(1:end-1) ], ...
    [ 0; reinforcement(1:end-1) ], ...
    [ 0; data{observations_to_fit(1:end-1), 'tdiff'} ], ...
    [ 0; 0; data{observations_to_fit(1:end-2), 'curkey'} ], ... %double lag curkey to prevent clairvoyance: only change curkey AFTER choice
    ]';

end
