    clear all;
    clc;
    
    NFFT = 256; %FFT number 
    Fs = 100;  % Sampling frequency/frame rate in Hz   
    num_of_std= 3; %number of standard deviation plus mean on the noise region to define the threshold
    imageThreshold = 66; %initial binary mask 
    
    fileDirectory = 'path/to/your/images/'; %set your file directory here
    fileName1= 'images'; $%file name prefix
    fileName2 = '.tif'; %file extension
   
    mkdir(strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\pure_freq\')); %set a folder for colored frequency map
    mkdir(strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\pure_freq_gray\')); %set a folder for grayscale frequency map
    mkdir(strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\complete_freq\')); %set a folder for colored frequency map together with scale bar
    mkdir(strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\hist\')); %set a folder for frequency histogram of the map
    mkdir(strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\Amp\')); %set a folder for fft amplitude of the map
    
    %read in the first frame to define the dimension of the frame
    frame=imread(strcat(fileDirectory,fileName1,num2str(1,'%04g'),fileName2));
    [f_height,f_length]=size(frame(:,:,1));
    
%you can edit numbers below based on your image series size and desired sliding window
for bas_num=0:256:(3000-256) % sliding window 
    
    for i = 1:NFFT
        %used im2gray instead of rgb2gray here since image is not rgb
         Vol(:,:,i)= im2gray(imread(strcat(fileDirectory,fileName1, num2str(bas_num+i,'%04g'),fileName2)));
    end 

    %first threshold mask
    avg=mean(Vol,3);
    zeroIndice = avg < imageThreshold;
    avg(zeroIndice) = 0;
    binaryImage = avg;
    oneIndice = avg >= imageThreshold ;
    binaryImage(oneIndice)=1;
    figure(2);
    imshow(binaryImage);
    %this two annotation could be unmarked to see the how the first binary mask works
    
                    
    f = Fs/2*linspace(0,1,NFFT/2+1); %index to frequency list
    timeProfile(1:f_height,1:f_length,1:NFFT) = 0;
    
    uppersignal= ceil(6*NFFT/Fs);
    lowersignal= ceil(1*NFFT/Fs);
    uppernoise= ceil(50*NFFT/Fs);
    lowernoise= ceil(6*NFFT/Fs);
    %this is to define signal region and noise region within the frequency domain
    
    timeProfile(:,:,:) = Vol(:,:,1:NFFT);
    timeProfileNoDC = timeProfile- mean(timeProfile,3);
    frequencyProfile =  fft(timeProfileNoDC,NFFT,3);
    
    frequencyAmplitudeProfile = (2*abs(frequencyProfile(:,:,1:NFFT/2+1)));
    
    
    spectralImage2D(:,:)=max(frequencyAmplitudeProfile(:,:,lowersignal:uppersignal),[],3).*binaryImage(:,:);
    thresholdMatrix(:,:)=binaryImage(:,:).*max(frequencyAmplitudeProfile(:,:,lowernoise:uppernoise),[],3);
     
    thresholdVector = reshape(thresholdMatrix,[f_height*f_length 1]);
    thresholdAmplitude = mean(thresholdVector(thresholdVector~=0))+ num_of_std*std(thresholdVector(thresholdVector~=0));

    spectralImage2D(spectralImage2D<=thresholdAmplitude) = 0;
    filteredSpectralImage2D = medfilt2(spectralImage2D, [4 4]);
    
    %this is to see the amplitude of the whole image 
    imwrite (mat2gray(filteredSpectralImage2D),strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\Amp\Amp_T',num2str(bas_num),'.tif'));
    
    zeroIndice = find(filteredSpectralImage2D <= 0);
    binaryImage(zeroIndice) = 0;
    oneIndice = find(filteredSpectralImage2D > 0 );
    binaryImage(oneIndice)=1;
    
    
    %this is to map cilia beat frequency
    frequencyImage(1:f_height,1:f_length) = 0;
    tarRegion(1:uppersignal-lowersignal+1) = 0;
    for o = 1:f_length
        for q = 1:f_height
            if binaryImage(q,o) == 1
               tarRegion(:) = frequencyAmplitudeProfile(q,o,lowersignal:uppersignal);
              
               P = find(tarRegion == max(tarRegion));
               frequencyImage(q,o) = (P(1)+lowersignal-2)*Fs/NFFT;
            else   
               frequencyImage(q,o) = 0;
          
            end    
        end
    end
    
    % this is to write the image in gray scale allow recoloring
    imwrite (mat2gray(frequencyImage),strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\pure_freq_gray\res',num2str(bas_num),'.tif'));
    
    % this is to write the image in parula color scale and set the lowest
    % color point to black
    orgimg1=ind2rgb(im2uint8(mat2gray(frequencyImage)),parula);
    redChannel1 = uint8(orgimg1(:, :, 1)*256);
    greenChannel1 = uint8(orgimg1(:, :, 2)*256);
    blueChannel1 = uint8(orgimg1(:, :, 3)*256);
    backgroundpixel = redChannel1 == 62 & greenChannel1  == 39 & blueChannel1  == 169;
    redChannel1(backgroundpixel) = 0;
    greenChannel1(backgroundpixel) = 0;
    blueChannel1(backgroundpixel) = 0;
    outputimg1 = cat(3, redChannel1, greenChannel1, blueChannel1);
    imwrite (outputimg1,strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\pure_freq\oct_pure_T',num2str(bas_num),'.tif'));
    
    % this is to output the image together with the color bar
    hFig=figure;
    imagesc(frequencyImage);
    colorbar;
    [cdata,colorMap]=getframe(hFig);
    redChannel = cdata(:, :, 1);
    greenChannel = cdata(:, :, 2);
    blueChannel = cdata(:, :, 3);
    backgroundpixel = redChannel == 61 & greenChannel  == 38 & blueChannel  == 168;
    redChannel(backgroundpixel) = 0;
    greenChannel(backgroundpixel) = 0;
    blueChannel(backgroundpixel) = 0;
    outputimg2 = cat(3, redChannel, greenChannel, blueChannel);
    imwrite(outputimg2,strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\complete_freq\oct_complete_T',num2str(bas_num),'.tif'));
    close(hFig);
    
   
    % this is to output the histogram of the frequency 
    f=figure;
    stretchfreq=reshape(frequencyImage,[f_height*f_length 1]);
    stretchfreq=stretchfreq(stretchfreq~=0);
    hist(stretchfreq,unique(stretchfreq));
    saveas(f, strcat(fileDirectory,fileName1,'\',num2str(NFFT),'f_',num2str(num_of_std),'std_threshold',num2str(imageThreshold),'\hist\hist_T',num2str(bas_num),'.tif'));
    close(f);
    
   
    end 