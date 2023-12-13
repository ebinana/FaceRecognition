%%
clear
clc
close all

%% Data extraction
% Training set
adr = './database/training1/';
fld = dir(adr);
nb_elt = length(fld);
% Data matrix containing the training images in its columns 
data_trn = []; 
% Vector containing the class of each training image
lb_trn = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_trn = [lb_trn ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_trn = [data_trn img(:)];
    end
end
% Size of the training set
[P,N] = size(data_trn);
% Classes contained in the training set
[~,I]=sort(lb_trn);
data_trn = data_trn(:,I);
[cls_trn,bd,~] = unique(lb_trn);
Nc = length(cls_trn); 
% Number of training images in each class
size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1]; 

%Display the database
F = zeros(192*Nc,168*max(size_cls_trn));
for i=1:Nc
    for j=1:size_cls_trn(i)
          pos = sum(size_cls_trn(1:i-1))+j;
          F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_trn(:,pos),[192,168]);
    end
end
figure;
imagesc(F);
colormap(gray);
axis off;

%% Parameters

[h,w,c]=size(imread("database\test1\yaleB01_P00A+005E+10.pgm"));
l=59;

%% Dataprocessing

x_bar=mean(data_trn,2);
data_centered=data_trn-x_bar;
X=data_centered/sqrt(N);
Gram=X'*X;
[V,D]=eig(Gram);
nonzero_eigenvalues = diag(D) > eps;
V_nonzero = V(:, nonzero_eigenvalues);
U = X*V_nonzero*diag(1./sqrt(diag(V_nonzero' * X' * X * V_nonzero)))*V_nonzero';

figure
for i = 1:Nc
    for j = 1:N/Nc
        k = (i-1)*(N/Nc) + j;
        subplot(Nc, N/Nc, k);
        imagesc(reshape(U(:,k), [h, w]));
        colormap('gray')
    end
end

E=zeros(1,N);
for i=1:N
    E(i)=(1/N)*sum(abs((U(:,i)')*(data_trn-x_bar)).^2);
end

[E,indexes]=sort(E,"descend");

kappaList=zeros(1,N-5);
for j=1:N-1
    kappaList(j)=calculate_kappa(U,X,j);
end


% x1_centered=data_trn(:,1)-x_bar;
% x1estim=sum(x1_centered*U(indexes(1:l)),2)+x_bar;
% figure,imagesc(abs(reshape(x1estim,[192,168]))),colormap('gray')
% 
% x2_centered=data_trn(:,11)-x_bar;
% x2estim=sum(x2_centered*U(indexes(1:l)),2)+x_bar;
% figure,imagesc(abs(reshape(x2estim,[192,168]))),colormap('gray')
% 
% x4_centered=data_trn(:,31)-x_bar;
% x4estim=sum(x4_centered*U(indexes(1:l)),2)+x_bar;
% figure,imagesc(abs(reshape(x4estim,[192,168]))),colormap('gray')




