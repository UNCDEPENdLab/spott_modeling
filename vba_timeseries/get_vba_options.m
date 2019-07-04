function [options, dim] = get_vba_options(data, vo)
%% this function sets up the options and dim structures for VBA

vo=validate_options(vo); %should be handled upstream, but just in case

options=[];
priors=[];

if ~vo.graphics
  options.DisplayWin = 0; %whether to display graphics during fitting
  options.GnFigs = 0;
end

% u is 2 x ntrials where first row is rt and second row is reward

% copy vo to inF and inG as starting point
options.inF = vo;
options.inG = vo;

% convergence settings
options.TolFun = 1e-6; %enforce a bit more precision
options.MaxIter = 256; %allow VBA to run for a while, if needed

options.GnTolFun = 1e-6;
options.GnMaxIter = 64;

options.verbose = 1; %don't show single subject fitting process

n_t = size(data,1); %number of rows

%% split into conditions/runs
if vo.multisession
  options.multisession.split = repmat(n_t/n_runs,1,n_runs);
  
  % fix parameters
  if fixed_params_across_runs
    options.multisession.fixed.theta = 'all';
    options.multisession.fixed.phi = 'all';
    
    % allow unique initial values for each run?
    options.multisession.fixed.X0 = 'all';
  end
end

%% skip first observation
options.skipf = zeros(1,n_t);
options.skipf(1) = 1;

%% specify dimensions of data to be fit
dim=struct();

dim = struct('n', vo.hidden_states, ... %number of hidden states
    'n_theta', vo.n_theta, ...
    'n_phi', vo.n_phi, ...
    'p', vo.n_outputs, ...
    'n_t', n_t);

% if ismember(vo.model, {'gvap', 'gvap_pmom', 'gvap_null', 'gvap_cross', 'gvap_hours', 'gvap_dayonly'})
%     dim = struct('n', vo.hidden_states, ... %number of hidden states
%         'n_theta', vo.n_theta, ...
%         'n_phi', vo.n_phi, ...
%         'p', vo.n_outputs, ...
%         'n_t', n_t);
% else
%     error('unknown model');
% end

%%populate priors
priors = get_priors(dim, vo);

options.priors = priors;
options.inG.priors = priors; %copy priors into inG for parameter transformation (e.g., Gaussian -> uniform)

options.sources(1) = struct('out', 1:2, 'type', 1); %choice is binomial

% if vo.multinomial
%   options.sources(1) = struct('out', 1:vo.ntimesteps, 'type', 2);
%   options.binomial = 1; %multinomial fitting
% end

end
