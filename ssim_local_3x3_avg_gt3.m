function out = ssim_local_3x3_avg_gt3(A, B)
% SSIM_LOCAL_3X3_AVG_GT3
% Local SIM/SIV/SIP using a 3×3 window. Statistics are computed as the
% average over *valid* cells only, and results are returned only when the
% number of valid cells in the window is > 3 (strictly greater).
%
% Constants: k1=0.01, k2=0.03, c1=(k1*R)^2, c2=(k2*R)^2, c3=c2/2.

A = double(A); B = double(B);
maskA  = isfinite(A);
maskB  = isfinite(B);
maskAB = maskA & maskB;

% 3×3 uniform kernel. The normalization cancels in ratios below, so we keep it.
W = ones(3,3)/9;
eps0 = 1e-12;

% Normalized-convolution helpers (NaN-safe via masks)
convN = @(X, M) conv2(X.*M, W, 'same');   % weighted sum over valid cells
convW = @(M)    conv2(M,   W, 'same');    % weighted count proxy

% -------- Local counts (strict >3 rule) --------
cntA  = convW(double(maskA))  * 9;   % convert back to raw counts
cntB  = convW(double(maskB))  * 9;
cntAB = convW(double(maskAB)) * 9;

% -------- Local means (average over valid cells only) --------
muA = convN(A, double(maskA)) ./ max(convW(double(maskA)), eps0);
muB = convN(B, double(maskB)) ./ max(convW(double(maskB)), eps0);
muA(cntA  <= 3) = NaN;   % keep only if >3 valid cells
muB(cntB  <= 3) = NaN;

% -------- Local variances (average over valid cells only) --------
varA = convN((A - muA).^2, double(maskA)) ./ max(convW(double(maskA)), eps0);
varB = convN((B - muB).^2, double(maskB)) ./ max(convW(double(maskB)), eps0);
varA(cntA <= 3) = NaN;
varB(cntB <= 3) = NaN;

% -------- Local covariance (average over joint-valid cells only) --------
% Use joint-support means for covariance
muAj = convN(A, double(maskAB)) ./ max(convW(double(maskAB)), eps0);
muBj = convN(B, double(maskAB)) ./ max(convW(double(maskAB)), eps0);
covAB = convN( (A - muAj).*(B - muBj), double(maskAB) ) ...
      ./ max(convW(double(maskAB)), eps0);
covAB(cntAB <= 3) = NaN;

% -------- Constants from data range --------
validVals = [A(maskA); B(maskB)];
R = max(validVals) - min(validVals);
if isempty(R) || ~isfinite(R) || R <= 0
    R = 1;
end

k1 = 0.01; k2 = 0.03;
c1 = (k1*R)^2;
c2 = (k2*R)^2;
c3 = c2/2;

% -------- Indices --------
sigmaA = sqrt(varA);
sigmaB = sqrt(varB);

SIM = (2.*muA.*muB + c1) ./ (muA.^2 + muB.^2 + c1);
SIV = (2.*sigmaA.*sigmaB + c2) ./ (varA + varB + c2);
SIP = (covAB + c3) ./ (sigmaA.*sigmaB + c3);

% -------- Global means --------
meanSIM = mean(SIM(:), 'omitnan');
meanSIV = mean(SIV(:), 'omitnan');
meanSIP = mean(SIP(:), 'omitnan');

% -------- Output --------
out = struct();
out.SIM = SIM; out.SIV = SIV; out.SIP = SIP;
out.meanSIM = meanSIM; out.meanSIV = meanSIV; out.meanSIP = meanSIP;
out.muA = muA; out.muB = muB; out.varA = varA; out.varB = varB; out.covAB = covAB;
out.kernel = W;
out.constants = struct('k1',k1,'k2',k2,'c1',c1,'c2',c2,'c3',c3,'R',R);
out.counts = struct('cntA',cntA,'cntB',cntB,'cntAB',cntAB);
end
