clear all
close all
clc

load phiAna.dat
load phiNum.dat


N.z = length(phiAna);
z = 0:20/(N.z-1):20;
plot(log(phiAna),z,'b',log(phiNum),z,'r--');
legend('analytical','numerical');
ylabel('z (m)')
xlabel('log(\phi) (V)')