clear;
close all;
taille_ecran = get(0,'ScreenSize');
L = taille_ecran(3);
H = taille_ecran(4);

% Parametres :
nb_iterations_GVF = 300;		% Nombre d'iterations du modele GVF
gamma_GVF = 0.01;
mu_GVF = 2;
nb_iterations_affichage = 20;

% Lecture et affichage de l'image a segmenter :
I = imread('pears.png');
[nb_lignes,nb_colonnes,nb_canaux] = size(I);
if nb_canaux==3
	I = rgb2gray(I);
end
I = double(I);
I = I/max(I(:));
figure('Name','Champ de force externe ameliore','Position',[0.05*L,0.05*H,0.9*L,0.7*H]);
subplot(1,2,1);
imagesc(I);
colormap gray;
axis image off;
axis xy;
title('Image a segmenter','FontSize',20);
drawnow;

% Champ de force externe de base :
[Ix,Iy] = gradient(I);
Eext_0 = -(Ix.*Ix+Iy.*Iy);
[Fx_0,Fy_0] = gradient(Eext_0);
norme_carre_Eext_0 = Fx_0.^2+Fy_0.^2;

% Champ de force externe du modele GVF :
Fx = Fx_0;
Fy = Fy_0;
for ii = 1:nb_iterations_GVF
	Lap_Fx = 4*del2(Fx);
	Lap_Fy = 4*del2(Fy);
	Fx = Fx-gamma_GVF*(norme_carre_Eext_0.*(Fx-Fx_0)-mu_GVF*Lap_Fx);
	Fy = Fy-gamma_GVF*(norme_carre_Eext_0.*(Fy-Fy_0)-mu_GVF*Lap_Fy);
	if mod(ii,nb_iterations_affichage)==1
		fprintf('Champ de force externe calcule a %d %%\n',round(100*ii/nb_iterations_GVF));
	end
end

% Normalisation du champ de force externe GVF pour l'affichage :
norme = sqrt(Fx.*Fx+Fy.*Fy);
Fx_normalise = Fx./(norme+eps);
Fy_normalise = Fy./(norme+eps);

% Affichage du champ de force externe du modele GVF :
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
title('Champ de force externe du modele GVF','FontSize',20);

save force_externe;
