#!/usr/bin/env python
# coding: utf-8

# In[25]:


from scipy.io import loadmat
data=loadmat("/Users/macbookair/Downloads/Data and Code/Dataset/UNMDataset/Jim_rest/826_1_PD_REST.mat")
print(data.keys())


# In[26]:


#EEG 변수 구조 확인
eeg_data = data['EEG']
print(type(eeg_data))
print(eeg_data.shape)


# In[27]:


#중첩된 구조 꺼내기
#MATLAB에서 저장된 방식(1x1 cell 구조처럼 저장됨)

eeg_inner= eeg_data[0,0]

print(type(eeg_inner))
print(eeg_inner)


# In[28]:


#저장된 EEG data는 '객체형'으로 저장되어 있으므로 EEG data와 채널 정보, 샘플링 속도를 직접 추출해야 함

#내부 필드 키 확인
print(dir(eeg_inner))


# In[29]:


#필드 확인 결과 numpy.void로 단순화돼서 필드 접근 불가능 => 데이터 다시 불러오기
from scipy.io import loadmat
data=loadmat('/Users/macbookair/Downloads/Data and Code/Dataset/UNMDataset/Jim_rest/826_1_PD_REST.mat',struct_as_record=False,squeeze_me=True)#구조체를 python 클래스처럼 사용 가능하게 함
eeg_inner=data['EEG']
print(type(eeg_inner))
print(dir(eeg_inner))


# In[30]:


#MNE가 필요로 하는 필드 선정,추출 후 MNE 변환
#수치 데이터 추출,샘플링 속도 추출,채널 이름 추출

#수치 데이터 추출
eeg_array = eeg_inner.data
print(eeg_array.shape)


# In[31]:


#샘플링 속도 추출
sfreq = eeg_inner.srate
print(sfreq)


# In[32]:


#채널 이름 추출
chanlocs = eeg_inner.chanlocs
print(chanlocs)


# In[33]:


print(type(eeg_inner.chanlocs))
print(len(eeg_inner.chanlocs))
print(type(eeg_inner.chanlocs[0]))
print(dir(eeg_inner.chanlocs[0]))


# In[34]:


#정확한 채널 이름 추출법
channel_names = [chan.labels for chan in eeg_inner.chanlocs]
print(channel_names)


# ## EEG Data 시각화 해석
# 1. 채널의 개수는 $67$개인데, 시각화 했을 때 $20$개만 보이는 이유?
# MNE는 기본적으로 채널 전체가 아닌 최대 $20$개의 채널만 보여주는 기본 설정값이 있음.
# 모든 채널을 다 볼 수 있게 할 수 있고, 또는 특정 채널만 추출해서 보여줄 수도 있음.
# 
# 2. 시각화 그래프가 왜 $2$개인 이유?
# 상단 그래프: EEG 시그널 상세 뷰
# 하단 그래프: 데이터 전체 타임라인 뷰(스크롤 바)

# In[35]:


#.m -> MNE Raw 객체 변환
import numpy as np
import mne

#수치 데이터 추출 (float32로 변환)
data_array = eeg_inner.data.astype(np.float32)

#샘플링 속도
sfreq = eeg_inner.srate

#채널 이름
channel_names = [chan.labels for chan in eeg_inner.chanlocs]

#채널 수만큼 'eeg' 타입 지정
ch_types = ['eeg']*len(channel_names)

#MNE Info 객체 만들기
info = mne.create_info(ch_names=channel_names,sfreq=sfreq,ch_types=ch_types)

#Raw 객체 생성
raw = mne.io.RawArray(data_array,info)

#데이터 시각화 확인
raw.plot()


# In[36]:


#전처리 
#1.리샘플링

raw_resampled = raw.copy().resample(250,npad="auto")
print(raw_resampled.info)


# In[37]:


#전처리
#2.밴드패스 필터링
raw_filtered = raw_resampled.copy().filter(2.5,14.,fir_design='firwin')
raw_filtered.plot()


# In[38]:


#오류 원인 파악 : ICA 단계 => 문제가 발생하는 채널은 좌표값이 아니라 채널 이름임.
print(channel_names[-5:])
print(data_array[-3,:10])
chan=eeg_inner.chanlocs[0]
print(f"Label: {chan.labels}, X: {chan.X}, Y: {chan.Y}, Z: {chan.Z}")


# In[39]:


#전처리
#3.ICA

from mne.preprocessing import ICA

#오류가 계속 생겼던 채널 제거
raw_filtered.drop_channels(['X','Y','Z'])

#veog 채널을 눈 깜빡임 채널(EOG)로 지정
#veog 채널이 눈 깜빡임 패턴을 가진 ICA 성분을 탐지할 때 기준이 되므로 아주 중요함!
raw_filtered.set_channel_types({'VEOG':'eog'})

#전극 위치 강제 지정
raw_filtered.set_montage('standard_1020')

ica = ICA(n_components=25,random_state=97,max_iter='auto')

ica.fit(raw_filtered)

ica.plot_components(inst=raw_filtered)


# In[40]:


#전처리
#3.ICA 
#성분 시각화 후 노이즈 파형 찾아 제거하기
ica.plot_sources(raw_filtered,show_scrollbars=True)


# In[41]:


#자동 탐지로 VEOG와 유사한 성분 찾기
#VEOG는 눈과 가까운 부분에서 측정한 신호이므로 눈 깜빡임에 대해 유의미한 지표를 제시할 것임.
eog_inds,eog_scores = ica.find_bads_eog(raw_filtered)
print(f"자동 탐지된 눈 깜빡임 성분 번호: {eog_inds}")


# In[20]:


#자동 탐지 결과 시각화
ica.plot_sources(raw_filtered,picks=eog_inds)


# In[42]:


#성분 제거 적용
ica.exclude = eog_inds
raw_cleaned = raw_filtered.copy()
ica.apply(raw_cleaned)


# In[ ]:




