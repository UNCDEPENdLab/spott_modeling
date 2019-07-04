function [phi_trans] = transform_phi(phi, inG)

if ismatrix(phi) && ~isvector(phi)
    asymm_check = abs(phi - phi') > 1e-5; %if ~issymmetric(phi) %this objects to tiny floating point imprecision??
    if any(asymm_check), error('Non-symmetric matrix passed to transform_phi'); end
    
    if ismember(inG.model, {'suuvid_base'})
        vars = diag(phi)'; %make sure this is a row vector
        sds_trans = sqrt([ exp(vars(1)), ... %beta: motor speed recovery rate must be positive
            exp(vars(2)), ... %gamma: positive
            vars(3), ... %nu: basal vigor allowed to be positive or negative -- keep as Gaussian
            gaminv(fastnormcdf(vars(4)), 2, 1), ... %kappa: Gamma(2,1) transform using inverse CDF approach. Use precompiled std norm cdf code for speed
            vars(5) ]); %stickiness can be positive or negative -- keep as Gaussian); %transform the parameters to get variances, then compute SDs
        phi_trans = transform_covmat(phi, sds_trans); %compute covariance matrix on rescaled parameters
    else
        phi_trans = phi; %just return untransformed for the moment...
    end
else
    
    if strcmpi(inG.model, 'suuvid_base')
        phi_trans = [ exp(phi(1)), ... %beta: motor speed recovery rate must be positive
            exp(phi(2)), ... %gamma: positive
            phi(3), ... %nu: basal vigor allowed to be positive or negative -- keep as Gaussian
            gaminv(fastnormcdf(phi(4)), 2, 1), ... %kappa: Gamma(2,1) transform using inverse CDF approach. Use precompiled std norm cdf code for speed
            phi(5) ... %stickiness can be positive or negative -- keep as Gaussian
            ];
    else
        error(['unrecognized model in transform_phi: ', inG.model]);
    end
end

end