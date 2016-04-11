close all
clear all
clc
beep off

N.x = 6;
N.y = 6;
N.z = 6;

for i=0:N.x-1
    for j=0:N.y-1
        for k=0:N.z-1
            text(i,j,k,num2str(i*N.y*N.z+ j*N.z + k + 1));
        end
    end
end
axis([-.5 N.x-.5 -.5 N.y-.5 -.5 N.z-.5]);
xlabel('x');
ylabel('y');
zlabel('z');
view(115,10);