% with the Buddha data, recovery is difficult
% and in fact most variations are (wrongly) computed as
% comming from the albedo.
load Buddha

nz = find(mask > 0);
[m,n,t] = size(I);

J = zeros(t, length(nz));
for i = 1 : t
    Ii = I(:,:,i);
    J(i,:) = Ii(nz);
end


% the easiest way to write it
% M = S\J;  
% the way that refers to the lecture
% my guess is that they are actually identical.
M = pinv(S)*J;
Rho = sqrt(M(1,:).^2 + M(2,:).^2 + M(3,:).^2);
N = M./repmat(Rho, [3 1]);
rho = zeros(m,n);
rho(nz) = Rho;

figure, imagesc(rho), axis image;

n1 = zeros(m,n);
n2 = zeros(m,n);
n3 = ones(m,n);
n1(nz) = N(1,:);
n2(nz) = N(2,:);
n3(nz) = N(3,:);

z = unbiased_integrate(n1, n2, n3, mask);
display_depth_mayavi(z)



