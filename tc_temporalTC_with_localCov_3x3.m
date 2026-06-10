function out = tc_temporalTC_with_localCov_3x3(X3, Y3, Z3, varargin)

% TC_2D_local 
% X3, Y3, Z3 are first normalized independently at each pixel:
%     Xn = (X - mean(X)) / std(X)
%     Yn = (Y - mean(Y)) / std(Y)
%     Zn = (Z - mean(Z)) / std(Z)
%
% TC errors and local spatial covariances are calculated using normalized
% anomaly fields. The final merged product is then taking X as the reference.

% -------- options
p = inputParser;
p.addParameter('Win', 3, @(x)isscalar(x)&&mod(x,2)==1);
p.addParameter('KminSpatial', 3, @(x)isscalar(x)&&x>=1);
p.addParameter('MinSamplesTime', 30, @(x)isscalar(x)&&x>=3);
p.addParameter('Eps', 1e-12, @(x)isscalar(x)&&x>0);
p.addParameter('SpaceMask', [], @(x)islogical(x)||isempty(x));
p.addParameter('EqualWeightsIfFail', true, @(x)islogical(x)||ismember(x,[0 1]));
p.parse(varargin{:});
opt = p.Results;

% -------- checks
[M,N,T] = size(X3);
assert(isequal(size(Y3),[M,N,T]) && isequal(size(Z3),[M,N,T]), ...
    'Inputs must be MxNxT.');

if isempty(opt.SpaceMask)
    spaceMask = true(M,N);
else
    spaceMask = logical(opt.SpaceMask);
end

K    = ones(opt.Win, opt.Win, 'double');
eps0 = opt.Eps;

%% =====================================================================
% (1) TEMPORAL ANOMALY NORMALIZATION
%     Each product is normalized independently at each pixel.
%% =====================================================================

Xn = nan(size(X3));
Yn = nan(size(Y3));
Zn = nan(size(Z3));

muX_all  = nan(M,N);
muY_all  = nan(M,N);
muZ_all  = nan(M,N);

stdX_all = nan(M,N);
stdY_all = nan(M,N);
stdZ_all = nan(M,N);

Ns_time = zeros(M,N);

for i = 1:M
    for j = 1:N

        if ~spaceMask(i,j)
            continue
        end

        x = squeeze(X3(i,j,:));
        y = squeeze(Y3(i,j,:));
        z = squeeze(Z3(i,j,:));

        m = isfinite(x) & isfinite(y) & isfinite(z);
        Ns_time(i,j) = nnz(m);

        if Ns_time(i,j) < opt.MinSamplesTime
            continue
        end

        x0 = x(m);
        y0 = y(m);
        z0 = z(m);

        muX  = mean(x0,'omitnan');
        muY  = mean(y0,'omitnan');
        muZ  = mean(z0,'omitnan');

        stdX = std(x0,0,'omitnan');
        stdY = std(y0,0,'omitnan');
        stdZ = std(z0,0,'omitnan');

        if stdX <= eps0 || stdY <= eps0 || stdZ <= eps0
            continue
        end

        muX_all(i,j)  = muX;
        muY_all(i,j)  = muY;
        muZ_all(i,j)  = muZ;

        stdX_all(i,j) = stdX;
        stdY_all(i,j) = stdY;
        stdZ_all(i,j) = stdZ;

        Xn(i,j,:) = (x - muX) ./ stdX;
        Yn(i,j,:) = (y - muY) ./ stdY;
        Zn(i,j,:) = (z - muZ) ./ stdZ;

    end
end

%% =====================================================================
% (2) TEMPORAL TC ERROR USING NORMALIZED ANOMALIES
%% =====================================================================

err_temporal = nan(M,N,3);

for i = 1:M
    for j = 1:N

        if ~spaceMask(i,j)
            continue
        end

        x = squeeze(Xn(i,j,:));
        y = squeeze(Yn(i,j,:));
        z = squeeze(Zn(i,j,:));

        m = isfinite(x) & isfinite(y) & isfinite(z);

        if nnz(m) >= opt.MinSamplesTime
            err_temporal(i,j,:) = tc_err_from_vectors(x(m), y(m), z(m), eps0);
        end

    end
end

%% =====================================================================
% (3) LOCAL SPATIAL COVARIANCES USING NORMALIZED ANOMALIES
%% =====================================================================

[cXX,cYY,cZZ,cXY,cXZ,cYZ, Nt_prop] = temporal_mean_local_covs_with_windowProps( ...
    Xn, Yn, Zn, K, opt.KminSpatial, spaceMask, eps0);

divsafe = @(num,den) (num ./ max(abs(den), eps0)) .* (abs(den) > eps0);

d2X_local = cXX - divsafe(cXY .* cXZ, cYZ);
d2Y_local = cYY - divsafe(cXY .* cYZ, cXZ);
d2Z_local = cZZ - divsafe(cXZ .* cYZ, cXY);

d2X_local = max(d2X_local, eps0);
d2Y_local = max(d2Y_local, eps0);
d2Z_local = max(d2Z_local, eps0);

err_local = cat(3, d2X_local, d2Y_local, d2Z_local);

%% =====================================================================
% (4) COMBINE TEMPORAL AND LOCAL ERRORS
%% =====================================================================

Ns_prop = Ns_time / T;
Ns_prop(~spaceMask) = NaN;
Nt_prop(~spaceMask) = NaN;

Err2D = nan(M,N,T,3);

for t = 1:T
    for i = 1:M
        for j = 1:N

            if ~spaceMask(i,j)
                continue
            end

            et = squeeze(err_temporal(i,j,:));
            es = squeeze(err_local(i,j,:));

            hasTemp  = all(isfinite(et));
            hasLocal = all(isfinite(es)) && isfinite(Nt_prop(i,j)) && Nt_prop(i,j) > 0;

            if hasTemp && hasLocal

                Nt = Nt_prop(i,j);
                Ns = Ns_prop(i,j);

                if isfinite(Nt) && isfinite(Ns) && (Nt + Ns) > 0
                    Err2D(i,j,t,:) = (Nt.*es.' + Ns.*et.') ./ (Nt + Ns);
                elseif isfinite(Ns)
                    Err2D(i,j,t,:) = et;
                elseif isfinite(Nt)
                    Err2D(i,j,t,:) = es;
                end

            end

        end
    end
end

%% =====================================================================
%% =====================================================================
% (5) MERGING IN NORMALIZED ANOMALY SPACE
%     Then back-transform merged anomaly to X absolute scale
%% =====================================================================

merged = nan(M,N,T);
merged_norm_all = nan(M,N,T);

wX     = nan(M,N);
wY     = nan(M,N);
wZ     = nan(M,N);
method = strings(M,N);

for i = 1:M
    for j = 1:N

        if ~spaceMask(i,j)
            continue
        end

        ebar = squeeze(mean(Err2D(i,j,:,:), 3, 'omitnan'));
        hasLocal2D = all(isfinite(ebar));

        if ~hasLocal2D

            x = squeeze(Xn(i,j,:));
            y = squeeze(Yn(i,j,:));
            z = squeeze(Zn(i,j,:));

            if any(isfinite(x) | isfinite(y) | isfinite(z))

                [Tmerge_norm_pix, ~] = tc_temporal_with_selection_var_pre( ...
                    x(:), y(:), z(:), ...
                    'minSamples', opt.MinSamplesTime, ...
                    'mergeMode', 'triple');

                merged_norm_all(i,j,:) = Tmerge_norm_pix(:);

                if isfinite(muX_all(i,j)) && isfinite(stdX_all(i,j))
                    merged(i,j,:) = Tmerge_norm_pix(:) .* stdX_all(i,j) + muX_all(i,j);
                end

                method(i,j) = "temporalTC";

            else
                method(i,j) = "none";
            end

            continue
        end

        invVar = 1 ./ max(ebar, eps0);
        ww = invVar ./ sum(invVar);

        wX(i,j) = ww(1);
        wY(i,j) = ww(2);
        wZ(i,j) = ww(3);
        method(i,j) = "local2D";

        for t = 1:T

            x = Xn(i,j,t);
            y = Yn(i,j,t);
            z = Zn(i,j,t);

            vals = [x; y; z];
            wv   = ww;

            ok = isfinite(vals) & isfinite(wv);

            if any(ok) && isfinite(muX_all(i,j)) && isfinite(stdX_all(i,j))

                wuse = wv(ok);
                wuse = wuse ./ sum(wuse);

                merged_norm = sum(wuse .* vals(ok));

                merged_norm_all(i,j,t) = merged_norm;

                % Back-transform to X/SMAP absolute soil moisture scale
                merged(i,j,t) = merged_norm .* stdX_all(i,j) + muX_all(i,j);

            end

        end

    end
end
%% =====================================================================
% Coverage diagnostics
%% =====================================================================

validMerged   = isfinite(merged) & repmat(spaceMask,1,1,T);
coverage_time = squeeze(sum(validMerged, [1 2])) ./ sum(spaceMask(:));
coverage_map  = mean(validMerged, 3, 'omitnan');

%% =====================================================================
% Outputs
%% =====================================================================

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

out.Xn            = Xn;
out.Yn            = Yn;
out.Zn            = Zn;

out.muX           = muX_all;
out.muY           = muY_all;
out.muZ           = muZ_all;

out.stdX          = stdX_all;
out.stdY          = stdY_all;
out.stdZ          = stdZ_all;

out.coverage_time = coverage_time;
out.coverage_map  = coverage_map;
out.method        = method;

end

%% =====================================================================
% Local covariance function
%% =====================================================================

function [cXX,cYY,cZZ,cXY,cXZ,cYZ,Nt_prop] = temporal_mean_local_covs_with_windowProps( ...
    X3, Y3, Z3, K, kmin, spaceMask, eps0)

[M,N,T] = size(X3);

cXXs=zeros(M,N); cYYs=zeros(M,N); cZZs=zeros(M,N);
cXYs=zeros(M,N); cXZs=zeros(M,N); cYZs=zeros(M,N);

nXX=zeros(M,N); nYY=zeros(M,N); nZZ=zeros(M,N);
nXY=zeros(M,N); nXZ=zeros(M,N); nYZ=zeros(M,N);

winOnes  = double(spaceMask);
Nt_accum = zeros(M,N);
Nt_n     = zeros(M,N);

for t = 1:T

    X = X3(:,:,t);
    Y = Y3(:,:,t);
    Z = Z3(:,:,t);

    mX = isfinite(X);
    mY = isfinite(Y);
    mZ = isfinite(Z);

    mXYZ = mX & mY & mZ & spaceMask;

    cntTriple = conv2(double(mXYZ), K, 'same');
    cntDen    = conv2(winOnes, K, 'same');

    propWin = cntTriple ./ max(cntDen,1);
    propWin(~isfinite(propWin)) = 0;

    Nt_accum = Nt_accum + propWin;
    Nt_n     = Nt_n + double(cntDen > 0);

    [vX, okX] = localCov2(X, X, mX, mX, spaceMask, K, kmin);
    fin = okX & isfinite(vX);
    cXXs(fin) = cXXs(fin) + vX(fin);
    nXX(fin)  = nXX(fin) + 1;

    [vY, okY] = localCov2(Y, Y, mY, mY, spaceMask, K, kmin);
    fin = okY & isfinite(vY);
    cYYs(fin) = cYYs(fin) + vY(fin);
    nYY(fin)  = nYY(fin) + 1;

    [vZ, okZ] = localCov2(Z, Z, mZ, mZ, spaceMask, K, kmin);
    fin = okZ & isfinite(vZ);
    cZZs(fin) = cZZs(fin) + vZ(fin);
    nZZ(fin)  = nZZ(fin) + 1;

    [c, ok] = localCov2(X, Y, mX, mY, spaceMask, K, kmin);
    fin = ok & isfinite(c);
    cXYs(fin) = cXYs(fin) + c(fin);
    nXY(fin)  = nXY(fin) + 1;

    [c, ok] = localCov2(X, Z, mX, mZ, spaceMask, K, kmin);
    fin = ok & isfinite(c);
    cXZs(fin) = cXZs(fin) + c(fin);
    nXZ(fin)  = nXZ(fin) + 1;

    [c, ok] = localCov2(Y, Z, mY, mZ, spaceMask, K, kmin);
    fin = ok & isfinite(c);
    cYZs(fin) = cYZs(fin) + c(fin);
    nYZ(fin)  = nYZ(fin) + 1;

end

cXX = cXXs ./ max(nXX,1); cXX(nXX==0) = NaN;
cYY = cYYs ./ max(nYY,1); cYY(nYY==0) = NaN;
cZZ = cZZs ./ max(nZZ,1); cZZ(nZZ==0) = NaN;

cXY = cXYs ./ max(nXY,1); cXY(nXY==0) = NaN;
cXZ = cXZs ./ max(nXZ,1); cXZ(nXZ==0) = NaN;
cYZ = cYZs ./ max(nYZ,1); cYZ(nYZ==0) = NaN;

Nt_prop = Nt_accum ./ max(Nt_n,1);
Nt_prop(Nt_n==0) = NaN;

end

%% =====================================================================
% Local covariance subfunction
%% =====================================================================

function [c, ok] = localCov2(A, B, mA, mB, spaceMask, K, kmin)

m   = (mA & mB) & spaceMask;
cnt = conv2(double(m), K, 'same');

ok  = cnt >= kmin;
den = cnt;
den(~ok) = NaN;

numA = conv2(A .* double(m), K, 'same');
muA  = nan(size(A));
muA(ok) = numA(ok) ./ den(ok);

numB = conv2(B .* double(m), K, 'same');
muB  = nan(size(B));
muB(ok) = numB(ok) ./ den(ok);

numAB = conv2((A .* B) .* double(m), K, 'same');
EAB   = nan(size(A));
EAB(ok) = numAB(ok) ./ den(ok);

c = EAB - muA .* muB;
c(~isfinite(c)) = NaN;

end

%% =====================================================================
% TC error from vectors
%% =====================================================================

function e = tc_err_from_vectors(x,y,z,eps0)

x = x(:) - mean(x,'omitnan');
y = y(:) - mean(y,'omitnan');
z = z(:) - mean(z,'omitnan');

C = cov([x y z], 1);

Cxx = C(1,1);
Cyy = C(2,2);
Czz = C(3,3);

Cxy = C(1,2);
Cxz = C(1,3);
Cyz = C(2,3);

divsafe = @(num,den) (num ./ max(abs(den), eps0)) .* (abs(den) > eps0);

d2X = Cxx - divsafe(Cxy * Cxz, Cyz);
d2Y = Cyy - divsafe(Cxy * Cyz, Cxz);
d2Z = Czz - divsafe(Cxz * Cyz, Cxy);

e = max([d2X d2Y d2Z], eps0);

end
