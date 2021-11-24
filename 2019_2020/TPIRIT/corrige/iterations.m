function [x,y] = iteration(x,y,Fx,Fy,gamma,A)

[nb_lignes,nb_colonnes] = size(Fx);

% Le calcul de la force externe necessite des coordonnees entieres :
i = round(y);
j = round(x); 
indices = sub2ind(size(Fx),i,j);

% Calcul de Bx et By :
Bx = -gamma*Fx(indices);
By = -gamma*Fy(indices);

% Mise a jour de x et y :
x = A*x+Bx;
y = A*y+By;	

