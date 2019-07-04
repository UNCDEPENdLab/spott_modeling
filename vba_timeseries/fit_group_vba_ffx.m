%loads in subjects' data and fits SUUVID models using VBA;

close all;
clear;
%curpath = fileparts(mfilename('fullpath'));

os = computer;
[~, me] = system('whoami');
me = strtrim(me);
is_alex=strcmp(me,'Alex')==1;

%note that this function looks for 'dataset' and 'model'
%as environment variables so that this script can be scaled easily for batch processing
%vo.model='decay_factorize_selective_psequate_fixedparams';

%% set environment and define file locations
project_repo = '~/Data_Analysis/spott_modeling';
data_source=[project_repo, '/data/vba_input'];
addpath(genpath('~/Documents/MATLAB/VBA-toolbox'));
addpath([project_repo, '/vba_timeseries/evo_functions']);

inputfiles = dir([data_source, '/*.csv']);

%extract IDs for record keeping
ids = cellfun(@(x) char(regexp(x,'[\d]+','match','once')), {inputfiles.name}, 'UniformOutput', false);

%convert inputs back into full paths
inputfiles = arrayfun(@(x) fullfile(x.folder, x.name), inputfiles, 'UniformOutput', false);

%exclude_ids = {'76'}; %A and P pretty much always at 100
%filter_vec = ~ismember(ids, exclude_ids);
%inputfiles = inputfiles(filter_vec);
%ids = ids(filter_vec);

%% setup parallel parameters
% ncpus=getenv('matlab_cpus');
% if strcmpi(ncpus, '')
%   ncpus=40;
%   fprintf('defaulting to 40 cpus because matlab_cpus not set\n');
% else
%   ncpus=str2double(ncpus);
% end
%

poolobj=parpool('local',4); %just use shared pool for now since it seems not to matter (no collisions)

%p = ProgressBar(length(inputfiles));
% models = {'ap', 'ap_ravg', 'ap_dayonly', 'ap_dayonly_nest', 'ap_hours', ...
%     'ap_null', 'ap_dynaffect', 'ap_dynaffect_hours', 'ap_dynaffect_hours_scalar', 'ap_dynaffect_homerun'};

models = {'suuvid_base'};

for mnum = 1:length(models)
    vo=[]; %vba options structure
    vo.model = models{mnum};
    vo.graphics = 1;
    vo = validate_options(vo); %initialize and validate suuvid fitting settings
        
    vo.output_dir = [project_repo, '/outputs/vba_out/ffx/', vo.model];
    if ~exist(vo.output_dir, 'dir'), mkdir(vo.output_dir); end

    % Log evidence matrix
    L = NaN(1,length(inputfiles));
    
    % Subject statistics cell vector
    s_all = cell(1,length(inputfiles));

    parfor sub = 1:length(inputfiles)
        o_file=sprintf('%s/fit_%s_%s_multisession%d', ...
            vo.output_dir, ids{sub}, vo.model, vo.multisession);
        
        fit_subj=1;
        if exist([o_file, '.mat'], 'file')
            m=matfile(o_file, 'writable', false);
            posterior=m.posterior;
            out=m.out;
            m=[]; %clear matfile handle
            fit_subj=0;
            fprintf('Skipping existing file: %s\n', o_file);
        else
            fprintf('Fitting subject %d id: %s \n', sub, ids{sub});
            [posterior, out] = suuvid_vba_fit_subject(inputfiles{sub}, vo);
        end
        
        s_all{sub} = extract_subject_statistics(posterior, out); %extract key statistics for each subject
        
        L(sub) = out.F;
        
        subj_id=ids{sub};
        
        %write subject data to mat file if just fitted
        if fit_subj == 1
            %parsave doesn't work in recent MATLAB versions.
            m=matfile(o_file, 'writable',true);
            m.posterior=posterior; m.out=out; m.subj_id=ids{sub}; m.subj_stats=s_all{sub};
        end
    end
    
    %save group outputs for now
    save(sprintf('%s/group_fits_%s_%s', vo.output_dir, vo.model, vo.dataset), 'ids', 'L', 'vo', 's_all');
    
    [group_global, group_trial_level] = extract_group_statistics(s_all, ...
        sprintf('%s/%s_%s_ffx_global_statistics.csv', vo.output_dir, vo.dataset, vo.model), ...
        sprintf('%s/%s_%s_ffx_prompt_outputs.csv', vo.output_dir, vo.dataset, vo.model));
    
    %save group outputs for now
    save(sprintf('%s/group_fits_%s_%s', vo.output_dir, vo.model, vo.dataset), 'ids', 'L', 'vo', 's_all', 'group_global', 'group_trial_level');
    
end

%p.stop;
delete(poolobj);


%doesn't work in recent MATLAB

 %parsave(sprintf('%s/fit_%s_%s_multinomial%d_multisession%d_fixedparams%d_uaversion%d', ...
        %		  vo.output_dir, ids{sub}, vo.model, vo.multinomial, vo.multisession, ...
        %    vo.fixed_params_across_runs, vo.u_aversion), posterior, out);%, subj_id);
       