clear; 
close all;
taille_ecran = get(0,'ScreenSize');
L = taille_ecran(3);
H = taille_ecran(4);

% Parametres :
sigma = 5;				% Ecart-type du noyau gaussien
T = ceil(3*sigma);			% Taille du noyau gaussien

% Lecture de l'image a segmenter :
I = imread('coins.png');
[nb_lignes,nb_colonnes,nb_canaux] = size(I);
if nb_canaux==3
	I = rgb2gray(I);
end
I = double(I);
I = I/max(I(:));
	
% Calcul et affichage de l'image filtree :
G = fspecial('Gaussian',[T T],sigma);	% Noyau gaussien de taille T x T et d'ecart-type sigma
Ig = conv2(I,G,'same');			% Filtrage gaussien par le noyau G
figure('Name','Amelioration du champ de force externe','Position',[0.05*L,0.05*H,0.9*L,0.7*H]);
subplot(1,2,1);
imagesc(Ig);
colormap gray;
axis image off;
axis xy;
title('Image a segmenter apres filtrage','FontSize',20);

% Nouveau champ de force externe :
[Igx,Igy] = gradient(Ig);
Eext = -(Igx.*Igx+Igy.*Igy);
[Fx,Fy] = gradient(Eext);

% Normalisation du nouveau champ de force externe pour l'affichage :
norme = sqrt(Fx.*Fx+Fy.*Fy);
Fx_normalise = Fx./(norme+eps);
Fy_normalise = Fy./(norme+eps);

% Affichage du nouveau champ de force externe :
subplot(1,2,2);
imagesc(I);
colormap gray;
axis image off;
axis xy;
hold on;
pas_fleches = 5;
taille_fleches = 1;
[x,y] = meshgrid(1:pas_fleches:nb_colonnes,1:pas_fleches:nb_lignes);
Fx_normalise_quiver = Fx_normalise(1:pas_fleches:nb_lignes,1:pas_fleches:nb_colonnes);
Fy_normalise_quiver = Fy_normalise(1:pas_fleches:nb_lignes,1:pas_fleches:nb_colonnes);
hq = quiver(x,y,Fx_normalise_quiver,Fy_normalise_quiver,taille_fleches);
set(hq,'LineWidth',1,'Color',[1,0,0]);
title('Champ de force externe ameliore','FontSize',20);

save force_externe;
