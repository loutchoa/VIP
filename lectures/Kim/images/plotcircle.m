function plotcircle(pos, radius, str)
    

v=[1,1];


for i=1:size(pos,1)
    rectangle('Position',[pos(i,:)-radius(i)*v, 2*radius(i)*v], ...
              'Curvature',[1,1], 'EdgeColor',str);
end
