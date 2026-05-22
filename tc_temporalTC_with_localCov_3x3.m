function out = tc_temporalTC_with_localCov_3x3(X3, Y3, Z3, varargin)
% TC_TEMPORALTC_WITH_LOCALCOV_3X3 (with temporal-TC fallback & method flag)
% Temporal Triple Collocation per pixel (over time) + Local (3x3) spatial
% covariances averaged over time. Rescales Y,Z -> X using TC-consistent
% ratios (use the third sensor to avoid attenuation). Combines temporal and
% local-spatial TC errors with per-pixel proportions (Ns, Nt) as in Eq.(9).
%
% Fallback:
%   - If LOCAL surroundings are not available (no valid local TC for this
%     pixel), the pixel is merged using the temporal selection TC:
%          tc_temporal_with_selection_var_pre(X(:,1),Y(:,1),Z(:,1),...)
%     applied to that pixel's time series.
%
% Method recording:
%   out.method(i,j) = "local2D"   → local temporal+spatial TC used
%                   = "temporalTC"→ fallback temporal selection TC used
%                   = "none"      → no merge (all NaN / insufficient data)
%
% Inputs
%   X3, Y3, Z3 : MxNxT numeric (NaNs allowed; aligned)
%
% Name-Value options
%   'Win'            : 3 (odd)  - local window size for spatial covariances
%   'KminSpatial'    : 2        - min joint-valid samples inside window (each time)
%   'MinSamplesTime' : 30       - min triple-overlap timesteps per pixel
%   'Eps'            : 1e-12    - small positive floor
%   'SpaceMask'      : []       - MxN logical study-area mask (1=in)
%   'EnforcePositiveScale' : true - force beta/gamma > 0
%   'EqualWeightsIfFail'   : true - (currently unused, kept for compatibility)
%
% Output (struct)
%   out.merged        : MxNxT merged field (units of X)
%   out.wX/Y/Z        : MxN time-constant weight maps (for local2D pixels)
%   out.err2D         : MxNxTx3 combined 2D errors [X Y Z] (only used where local2D)
%   out.err_temporal  : MxNx3 temporal TC errors
%   out.err_local     : MxNx3 local-spatial TC errors (from temporal-mean local covs)
%   out.Ns_prop       : MxN per-pixel temporal proportion (Ns/T)
%   out.Nt_prop       : MxN per-pixel local-window proportion
%   out.betaY, out.betaZ : MxN temporal rescale factors (Y,Z->X)
%   out.coverage_time : 1xT fraction of in-mask pixels with valid merged
%   out.coverage_map  : MxN fraction of time steps with valid merged
%   out.method        : MxN string map of which merge method used
%   out.notes         : options & bookkeeping

% -------- options
p = inputParser;
p.addParameter('Win', 3, @(x)isscalar(x)&&mod(x,2)==1);
p.addParameter('KminSpatial', 2, @(x)isscalar(x)&&x>=1);
p.addParameter('MinSamplesTime', 30, @(x)isscalar(x)&&x>=3);
p.addParameter('Eps', 1e-12, @(x)isscalar(x)&&x>0);
p.addParameter('SpaceMask', [], @(x)islogical(x)||isempty(x));
p.addParameter('EnforcePositiveScale', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('EqualWeightsIfFail', true, @(x)islogical(x)||ismember(x,[0 1]));
p.parse(varargin{:});
opt = p.Results;

% -------- checks
[M,N,T] = size(X3);
assert(isequal(size(Y3),[M,N,T]) && isequal(size(Z3),[M,N,T]), 'Inputs must be MxNxT.');
if isempty(opt.SpaceMask), spaceMask = true(M,N); else, spaceMask = logical(opt.SpaceMask); end
K    = ones(opt.Win, opt.Win, 'double');
eps0 = opt.Eps;

% =====================================================================
% (1) TEMPORAL TC with TC-consistent RESCALING (per pixel)
%     betaY = Cov_t(X,Z)/Cov_t(Y,Z),  betaZ = Cov_t(X,Y)/Cov_t(Z,Y)
% =====================================================================
betaY   = nan(M,N); 
betaZ   = nan(M,N);
Ns_time = zeros(M,N);  % temporal triple-overlap counts

for i = 1:M
    for j = 1:N
        if ~spaceMask(i,j), continue; end
        x = squeeze(X3(i,j,:));
        y = squeeze(Y3(i,j,:));
        z = squeeze(Z3(i,j,:));
        m = isfinite(x) & isfinite(y) & isfinite(z);
        Ns_time(i,j) = nnz(m);
        if Ns_time(i,j) >= opt.MinSamplesTime
            C = cov([x(m), y(m), z(m)], 1);      % [X Y Z] across time
            % TC-consistent scale ratios:
            %   betaY = Cov(X,Z)/Cov(Y,Z), betaZ = Cov(X,Y)/Cov(Z,Y)
            if abs(C(2,3)) > eps0, betaY(i,j) = C(1,3) / C(2,3); end
            if abs(C(3,2)) > eps0, betaZ(i,j) = C(1,2) / C(3,2); end
        end
    end
end

if opt.EnforcePositiveScale
    betaY(betaY <= 0) = NaN;
    betaZ(betaZ <= 0) = NaN;
end
betaY(~isfinite(betaY)) = 1;
betaZ(~isfinite(betaZ)) = 1;

% Rescaled series for TC-consistent local 2D work
Yr = Y3; 
Zr = Z3;
for t = 1:T
    Yr(:,:,t) = betaY .* Y3(:,:,t);
    Zr(:,:,t) = betaZ .* Z3(:,:,t);
end

% ---- temporal TC error per pixel on rescaled data
err_temporal = nan(M,N,3);  % [X Y Z]
for i = 1:M
    for j = 1:N
        if ~spaceMask(i,j), continue; end
        x = squeeze(X3(i,j,:));
        y = squeeze(Yr(i,j,:));
        z = squeeze(Zr(i,j,:));
        m = isfinite(x) & isfinite(y) & isfinite(z);
        if nnz(m) >= opt.MinSamplesTime
            err_temporal(i,j,:) = tc_err_from_vectors(x(m), y(m), z(m), eps0);
        end
    end
end

% =====================================================================
% (2) LOCAL (3x3) SPATIAL COVARIANCES per time -> TEMPORAL MEAN
%     Then TC errors from those local covs (spatial component)
% =====================================================================
[cXX,cYY,cZZ,cXY,cXZ,cYZ, Nt_prop] = temporal_mean_local_covs_with_windowProps( ...
    X3, Yr, Zr, K, opt.KminSpatial, spaceMask, eps0);

% spatial/local TC error variances (time-constant maps)
divsafe   = @(num,den) (num ./ max(abs(den), eps0)) .* (abs(den) > eps0);
d2X_local = cXX - divsafe(cXY .* cXZ, cYZ);
d2Y_local = cYY - divsafe(cXY .* cYZ, cXZ);
d2Z_local = cZZ - divsafe(cXZ .* cYZ, cXY);
d2X_local = max(d2X_local, eps0);
d2Y_local = max(d2Y_local, eps0);
d2Z_local = max(d2Z_local, eps0);
err_local = cat(3, d2X_local, d2Y_local, d2Z_local);   % MxNx3

% =====================================================================
% (3) 2D ERROR via Eq.(9) where LOCAL exists; leave NaN elsewhere
% =====================================================================
Ns_prop = Ns_time / T;                    % per-pixel temporal proportion
Ns_prop(~spaceMask) = NaN;
Nt_prop(~spaceMask) = NaN;

Err2D = nan(M,N,T,3);  % only meaningful where both temporal & local exist

for t = 1:T
    for i = 1:M
        for j = 1:N
            if ~spaceMask(i,j), continue; end

            et = squeeze(err_temporal(i,j,:));    % 3×1 temporal TC errors
            es = squeeze(err_local(i,j,:));       % 3×1 local TC errors (time-constant)

            hasTemp  = all(isfinite(et));
            hasLocal = all(isfinite(es)) && isfinite(Nt_prop(i,j)) && (Nt_prop(i,j) > 0);

            if hasTemp && hasLocal
                % ---- FULL 2D TC (temporal + local) ----
                Nt = Nt_prop(i,j);
                Ns = Ns_prop(i,j);
                if isfinite(Nt) && isfinite(Ns) && (Nt + Ns) > 0
                    Err2D(i,j,t,:) = (Nt.*es.' + Ns.*et.') ./ (Nt + Ns);
                elseif isfinite(Ns)
                    % Only temporal counts meaningful -> temporal-only
                    Err2D(i,j,t,:) = et;
                elseif isfinite(Nt)
                    % Only spatial-window counts meaningful -> local-only
                    Err2D(i,j,t,:) = es;
                end
            else
                % No valid local surroundings or no temporal TC -> leave NaN.
                % These pixels will use temporal selection TC downstream.
            end
        end
    end
end

% =====================================================================
% (4) PER-PIXEL WEIGHTS FOR LOCAL 2D TC (time-constant) & MERGING
%     If no local 2D available → call tc_temporal_with_selection_var_pre()
% =====================================================================
merged = nan(M,N,T);
wX     = nan(M,N);
wY     = nan(M,N);
wZ     = nan(M,N);
method = strings(M,N);   % "local2D" / "temporalTC" / "none"

for i = 1:M
    for j = 1:N
        if ~spaceMask(i,j), continue; end

        % Check if this pixel has valid local 2D TC info at least at some times
        ebar = squeeze(mean(Err2D(i,j,:,:), 3, 'omitnan'));  % 3×1
        hasLocal2D = all(isfinite(ebar));

        if ~hasLocal2D
            % ==========================================================
            % FALLBACK: temporal selection TC for this pixel
            % ==========================================================
            x = squeeze(X3(i,j,:));
            y = squeeze(Y3(i,j,:));
            z = squeeze(Z3(i,j,:));

            % Only call if there is at least some data
            if any(isfinite(x) | isfinite(y) | isfinite(z))
                [Tmerge_pix, ~] = tc_temporal_with_selection_var_pre( ...
                    x(:), y(:), z(:), ...
                    'minSamples', opt.MinSamplesTime, ...
                    'mergeMode', 'triple');
                % Tmerge_pix: T×1
                merged(i,j,:) = Tmerge_pix(:);
                method(i,j)   = "temporalTC";
            else
                method(i,j)   = "none";
            end

            continue;
        end

        % ==============================================================
        % LOCAL 2D TC branch: use Err2D to define BLUE weights, then
        % merge using rescaled Y,Z (Yr,Zr). Weights are time-constant.
        % ==============================================================
        invVar = 1 ./ max(ebar, eps0);
        ww     = invVar ./ sum(invVar);   % 3×1

        wX(i,j) = ww(1);
        wY(i,j) = ww(2);
        wZ(i,j) = ww(3);
        method(i,j) = "local2D";

        % Merge timeslices (availability-aware: any subset of X,Y,Z)
        for t = 1:T
            x = X3(i,j,t);
            y = Yr(i,j,t);
            z = Zr(i,j,t);

            vals = [x; y; z];
            wv   = ww;

            ok = isfinite(vals) & isfinite(wv);
            if all(ok)
                wuse = wv(ok);
                wuse = wuse / sum(wuse);
                merged(i,j,t) = sum(wuse .* vals(ok));
            end
        end
    end
end

% =====================================================================
% Coverage diagnostics
% =====================================================================
validMerged   = isfinite(merged) & repmat(spaceMask,1,1,T);
coverage_time = squeeze(sum(validMerged, [1 2])) ./ sum(spaceMask(:)); % 1×T
coverage_map  = mean(validMerged, 3, 'omitnan');                       % M×N

% =====================================================================
% outputs
% =====================================================================
out = struct();
out.merged        = merged;
out.wX            = wX;
out.wY            = wY;
out.wZ            = wZ;
out.err2D         = Err2D;
out.err_temporal  = err_temporal;
out.err_local     = err_local;
out.Ns_prop       = Ns_prop;
out.Nt_prop       = Nt_prop;
out.betaY         = betaY;
out.betaZ         = betaZ;
out.coverage_time = coverage_time;
out.coverage_map  = coverage_map;
out.method        = method;      % "local2D" / "temporalTC" / "none"
out.notes.options = opt;

end

% ---------------------------------------------------------------------
function [cXX,cYY,cZZ,cXY,cXZ,cYZ,Nt_prop] = temporal_mean_local_covs_with_windowProps( ...
    X3, Y3, Z3, K, kmin, spaceMask, eps0)
% TEMPORAL_MEAN_LOCAL_COVS_WITH_WINDOWPROPS  (NaN-safe, proper ends)
%
% For each time: compute local (co)variances inside a W×W window,
% then take temporal means. Also return mean triple-valid proportion Nt_prop.

[M,N,T] = size(X3);

cXXs=zeros(M,N); cYYs=zeros(M,N); cZZs=zeros(M,N);
cXYs=zeros(M,N); cXZs=zeros(M,N); cYZs=zeros(M,N);
nXX =zeros(M,N); nYY =zeros(M,N); nZZ =zeros(M,N);
nXY =zeros(M,N); nXZ =zeros(M,N); nYZ =zeros(M,N);

winOnes  = double(spaceMask);
Nt_accum = zeros(M,N); 
Nt_n     = zeros(M,N);

for t = 1:T
    X = X3(:,:,t); Y = Y3(:,:,t); Z = Z3(:,:,t);
    mX = isfinite(X); mY = isfinite(Y); mZ = isfinite(Z);
    mXYZ = mX & mY & mZ & spaceMask;

    % ---- triple-valid proportion (coverage) for this time
    cntTriple = conv2(double(mXYZ), K, 'same');
    cntDen    = conv2(winOnes,       K, 'same');
    propWin   = cntTriple ./ max(cntDen,1);
    propWin(~isfinite(propWin)) = 0;
    Nt_accum = Nt_accum + propWin;
    Nt_n     = Nt_n + double(cntDen>0);

    % ---- local variances
    [vX, okX] = localCov2(X, X, mX, mX, spaceMask, K, kmin);
    fin = okX & isfinite(vX);
    cXXs(fin) = cXXs(fin) + vX(fin);  nXX(fin) = nXX(fin) + 1;

    [vY, okY] = localCov2(Y, Y, mY, mY, spaceMask, K, kmin);
    fin = okY & isfinite(vY);
    cYYs(fin) = cYYs(fin) + vY(fin);  nYY(fin) = nYY(fin) + 1;

    [vZ, okZ] = localCov2(Z, Z, mZ, mZ, spaceMask, K, kmin);
    fin = okZ & isfinite(vZ);
    cZZs(fin) = cZZs(fin) + vZ(fin);  nZZ(fin) = nZZ(fin) + 1;

    % ---- local covariances
    [c, ok] = localCov2(X, Y, mX, mY, spaceMask, K, kmin);
    fin = ok & isfinite(c);
    cXYs(fin) = cXYs(fin) + c(fin);   nXY(fin) = nXY(fin) + 1;

    [c, ok] = localCov2(X, Z, mX, mZ, spaceMask, K, kmin);
    fin = ok & isfinite(c);
    cXZs(fin) = cXZs(fin) + c(fin);   nXZ(fin) = nXZ(fin) + 1;

    [c, ok] = localCov2(Y, Z, mY, mZ, spaceMask, K, kmin);
    fin = ok & isfinite(c);
    cYZs(fin) = cYZs(fin) + c(fin);   nYZ(fin) = nYZ(fin) + 1;
end

% ---- temporal means (NaN where no valid times)
cXX = cXXs ./ max(nXX,1); cXX(nXX==0) = NaN;
cYY = cYYs ./ max(nYY,1); cYY(nYY==0) = NaN;
cZZ = cZZs ./ max(nZZ,1); cZZ(nZZ==0) = NaN;
cXY = cXYs ./ max(nXY,1); cXY(nXY==0) = NaN;
cXZ = cXZs ./ max(nXZ,1); cXZ(nXZ==0) = NaN;
cYZ = cYZs ./ max(nYZ,1); cYZ(nYZ==0) = NaN;

% ---- mean proportion of triple-valid samples
Nt_prop = Nt_accum ./ max(Nt_n,1);
Nt_prop(Nt_n==0) = NaN;

end

% =====================================================================
% Local subfunction
% =====================================================================
function [c, ok] = localCov2(A, B, mA, mB, spaceMask, K, kmin)
% Compute local covariance/variance between A and B inside window K
% NaN-safe; only returns finite values where joint-valid count >= kmin.

m   = (mA & mB) & spaceMask;
cnt = conv2(double(m), K, 'same');

ok  = cnt >= kmin;
den = cnt;  den(~ok) = NaN;                 % avoid divide-by-zero; mark invalid windows

% local means (only valid where ok)
numA = conv2(A .* double(m), K, 'same');
muA  = nan(size(A)); muA(ok) = numA(ok) ./ den(ok);
numB = conv2(B .* double(m), K, 'same');
muB  = nan(size(B)); muB(ok) = numB(ok) ./ den(ok);

% local E[AB]
numAB = conv2((A.*B) .* double(m), K, 'same');
EAB   = nan(size(A)); EAB(ok) = numAB(ok) ./ den(ok);

% covariance/variance
c = EAB - muA .* muB;
c(~isfinite(c)) = NaN;

end

% ---------------------------------------------------------------------
function e = tc_err_from_vectors(x,y,z,eps0)
% TC error variances from 3 vectors (population covariance), NaN-safe.
x = x(:)-mean(x,'omitnan');
y = y(:)-mean(y,'omitnan');
z = z(:)-mean(z,'omitnan');
C = cov([x y z], 1);
Cxx=C(1,1); Cyy=C(2,2); Czz=C(3,3);
Cxy=C(1,2); Cxz=C(1,3); Cyz=C(2,3);
divsafe = @(num,den) (num ./ max(abs(den), eps0)) .* (abs(den)>eps0);
d2X = Cxx - divsafe(Cxy*Cxz, Cyz);
d2Y = Cyy - divsafe(Cxy*Cyz, Cxz);
d2Z = Czz - divsafe(Cxz*Cyz, Cxy);
e   = max([d2X d2Y d2Z], eps0);
end
