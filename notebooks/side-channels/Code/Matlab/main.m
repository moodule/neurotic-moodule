%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% find the root path
rootdir = fileparts(which('main'));
cd(rootdir);

[Y,FS,NBITS] = wavread('../../Data/con_11.wav');
n = size(Y,1);
t = 1:n;

plot(t,Y(:,1));

%[Y,FS,NBITS]=wavread('..\..\Data\test.wav')
