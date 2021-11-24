load Beethoven

nz = find(mask > 0);
[m,n] = size(mask);

J = zeros(3, length(nz));
for i = 1 : 3
    Ii = I(:,:,i);
    J(i,:) = Ii(nz);
end

M = S\J;
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
display_depth(z)



