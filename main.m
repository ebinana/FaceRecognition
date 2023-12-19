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

% %Display the database
% F = zeros(192*Nc,168*max(size_cls_trn));
% for i=1:Nc
%     for j=1:size_cls_trn(i)
%           pos = sum(size_cls_trn(1:i-1))+j;
%           F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_trn(:,pos),[192,168]);
%     end
% end
% figure;
% imagesc(F);
% colormap(gray);
% axis off;

%% Parameters

[h,w,c]=size(imread("database\test1\yaleB01_P00A+005E+10.pgm"));
alpha=0.9;

%% Dataprocessing

x_bar=mean(data_trn,2);
data_centered=data_trn-x_bar;
X=data_centered/sqrt(N);
Gram=X'*X;
[V,D]=eig(Gram);
[eiglist,indexes]=sort(diag(D),"descend");
eiglist = eiglist(1:N-1);
indexes=indexes(1:N-1)-1;
V = V(:, indexes);
U = X*V*((V'*Gram*V)^(-1/2));


%%% Calcul de l
kappaList=zeros(1,N-5);
for j=1:N-1
    kappaList(j)=sum(eiglist(1:j))/sum(eiglist);
end

l = sum(kappaList<alpha);
% l=5;

%%%

%%% affichage des eigenfaces
% figure
% for i = 1:Nc
%     if (i==Nc)
%         for j = 1:N/Nc-1
%             k = (i-1)*(N/Nc) + j;
%             subplot(Nc, N/Nc, k);
%             imagesc(reshape(U(:,k), [h, w]));
%             colormap('gray')
%         end
%     else
%         for j = 1:N/Nc
%             k = (i-1)*(N/Nc) + j;
%             subplot(Nc, N/Nc, k);
%             imagesc(reshape(U(:,k), [h, w]));
%             colormap('gray')
%         end
%     end
% end
% %%% 
% 
% %%% Affichage visage reconstruit
% 
% x1_centered=data_trn(:,1)-x_bar;
% tmp1=x1_centered'*U(:,1:l);
% out=zeros(length(x1_centered),l);
% for i = 1:l 
%     out(:,i)=tmp1(i)*U(:,i);
% end
% x1estim=sum(out,2)+x_bar;
% figure, 
% subplot 121
% imagesc(reshape(x1estim,[192,168])),colormap('gray')
% subplot 122
% imagesc(reshape(data_trn(:,1),[192,168])),colormap('gray')

%%% Classification

test=KNN(5,data_trn(:,13),data_trn,U,l);
test=floor((test-1)/10)+1;

% Trouver la valeur la plus fréquente
valeurs_uniques = unique(test);  % Permet d'avoir une liste des valeurs apparaissant une fois
occurrences = hist(test, valeurs_uniques); % Calcul le nombre de fois que l'elt apparait dans le vecteur
[~, index] = max(occurrences);   %trouver l'index de l'elt qui apparait le + de fois
valeur_la_plus_presente = valeurs_uniques(index);  % prendre la valeur


