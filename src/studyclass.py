
#load dependencies
from sql_conn import connection #SQL connection details and variables
import neuralnet as nn #functions to form neural network and calculate accuracy and plot accuracy with history
import preproc as pr #functions to preprocess data 
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing #for one-hot encoding

#get data
#Variable s has SQL script for fetching data from ODBC connection
s="SELECT IC.ICDBlock, SS.StudyLeadAdmin_NIHRDev, SS.StudyLeadAdmin, SS.StudyLeadAdminOrDivision, SS.StudyLeadAdminComm, SS.StudyManagingRDO, SS.StudyManagingSpecialtyId, SS.StudyManagingSpecialty, SS.StudyLeadLCRN, SS.StudyShortName, SS.StudyTitle, SS.StudyPriorityPandemic, SS.StudyPortfolioEligibility, SS.StudyGovernanceRoute, SS.StudyManagingSpecialty_PrimarySubSpecialty, SS.StudySubSpecialties_Concatenated, SS.StudyHRCSHealthCategories_Concatenated, SS.StudyRandomisationStatus, SS.StudyDesignType, SS.StudyIntervention, SS.StudyDesignTypeIntOb, SS.StudyInterventionDetail_Concatenated, SS.StudyPhases_Concatenated, SS.StudySettings_Concatenated, SS.StudyGeographicalScope, SS.StudyCommercialInvolvement, SS.StudyCTU, SS.StudySampleSizeUK, SS.StudySampleSizeGlobal, SS.StudySampleSizeNI, SS.StudyManagingRDOShort, SS.StudySampleSizeWales, SS.StudySampleSizeEngland, SS.StudyHasScreeningElement, SS.StudyShouldUploadRecruitmentData, SS.StudyMainFunder, SS.StudyHasMainFunder, SS.StudyFunders_Concatenated, SS.StudySponsor, SS.StudySponsorType, SS.StudyCROthenSponsor, SS.StudyInclusionCriteria, SS.StudyExclusionCriteria, SS.StudyResearchSummary, SS.StudyCommercialContributionDetail_Concatenated, SS.StudyConsumerInvolvement, SS.StudyConsumerInvolvementDetail, SS.Study_IsUrgentPublicHealthResearch, SS.Study_IsExperimentalMedicineComponent, SS.Study_IsCommercial, SS.Study_IsCTU, SS.StudyCountDraftAndLive, SS.Study_IsPostApril2010, SS.Study_IsInPublicSearch, SS.Study_IsHRCSICDReviewed, SS.StudyComplexityCategory, SS.StudyCommercialOrCollaborative, SS.StudyCommercialStudy, SS.Study_IsOpen, SS.Study_IsClosed, SS.StudyCount, SS.StudyResearchCategory, SS.StudyCommercialStudyType, SS.StudyFeasibilityStatus, SS.StudyMDSComplete, SS.Study_IsJDRStudy, SS.StudySampleSizeScotland, SS.Study_IsDraft, SS.Study_IsCROStudy, SS.StudyResearchAssessment, SS.StudyObservationalDetail_Concatenated, SS.StudyComplexityCategoryExtra, SS.StudyPriority, SS.Study_IsNonNHS, SS.StudyICDCodingRequired, SS.StudyComplexityCategoryWeighting, SS.StudyLeadLCRNAcronym, SS.StudyLeadLCRNMedium, SS.StudyComplexityCategoryLargeOther, SS.StudyUpperAgeLimit, SS.StudyLowerAgeLimit, SS.Study_PandemicStatusConfirmed, SS.Study_IsManagedRecovery, SS.Study_IsCOVID, SS.Study_DoesStudyRecordParticipantAge from (SELECT * FROM [CPMS_BI].dbo.[ODP_Study] WHERE StudyICDCodingRequired='Yes') AS SS LEFT JOIN (SELECT * from [CPMS_BI].[dbo].[ODP_StudyICD] where ICDBlock = 'Other forms of heart disease') AS IC ON SS.StudyID = IC.StudyID" 
                                    
studyICDMAP = pd.read_sql(s, connection)
#clean data
#subset data
pd_df=studyICDMAP[['ICDBlock','StudyLeadAdmin_NIHRDev','StudyLeadAdmin','StudyLeadAdminOrDivision','StudyLeadAdminComm','StudyManagingRDO','StudyManagingSpecialty','StudyLeadLCRN','StudyShortName','StudyTitle','StudyPriorityPandemic','StudyPortfolioEligibility','StudyGovernanceRoute','StudyManagingSpecialty_PrimarySubSpecialty','StudySubSpecialties_Concatenated','StudyHRCSHealthCategories_Concatenated','StudyRandomisationStatus','StudyDesignType','StudyIntervention','StudyDesignTypeIntOb','StudyInterventionDetail_Concatenated','StudyPhases_Concatenated','StudySettings_Concatenated','StudyGeographicalScope','StudyCommercialInvolvement','StudyCTU','StudySampleSizeUK','StudySampleSizeGlobal','StudySampleSizeNI','StudyManagingRDOShort','StudySampleSizeWales','StudySampleSizeEngland','StudyHasScreeningElement','StudyShouldUploadRecruitmentData','StudyMainFunder','StudyHasMainFunder','StudyFunders_Concatenated','StudySponsor','StudySponsorType','StudyCROthenSponsor','StudyInclusionCriteria','StudyExclusionCriteria','StudyResearchSummary','StudyCommercialContributionDetail_Concatenated','StudyConsumerInvolvement','StudyConsumerInvolvementDetail','Study_IsUrgentPublicHealthResearch','Study_IsExperimentalMedicineComponent','Study_IsCommercial','Study_IsCTU','StudyCountDraftAndLive','Study_IsPostApril2010','Study_IsInPublicSearch','Study_IsHRCSICDReviewed','StudyComplexityCategory','StudyCommercialOrCollaborative','StudyCommercialStudy','Study_IsOpen','Study_IsClosed','StudyCount','StudyResearchCategory','StudyCommercialStudyType','StudyFeasibilityStatus','StudyMDSComplete','Study_IsJDRStudy','StudySampleSizeScotland','Study_IsDraft','Study_IsCROStudy','StudyResearchAssessment','StudyObservationalDetail_Concatenated','StudyComplexityCategoryExtra','StudyPriority','Study_IsNonNHS','StudyICDCodingRequired','StudyComplexityCategoryWeighting','StudyLeadLCRNAcronym','StudyLeadLCRNMedium','StudyComplexityCategoryLargeOther','StudyUpperAgeLimit','StudyLowerAgeLimit','Study_PandemicStatusConfirmed','Study_IsManagedRecovery','Study_IsCOVID','Study_DoesStudyRecordParticipantAge']]
#remove text fields - also removed concatenated study specialty, and other concatenated fields, HRCS_Concatenated removed, 'StudyCTU' removed,
#removed - 'StudyCROthenSponsor','StudyInclusionCriteria','StudyExclusionCriteria','StudyResearchSummary'
#removed - study sponsor, 'StudyMainFunder', funders concatenated, 'StudyCommercialContributionDetail_Concatenated'
#removed - 'Study_IsOpen','Study_IsClosed', 'StudyLeadLCRNMedium', 'StudyICDCodingRequired'
#removed - 'StudyObservationalDetail_Concatenated'
data=pd_df[['ICDBlock','StudyLeadAdmin_NIHRDev','StudyLeadAdmin','StudyLeadAdminOrDivision','StudyLeadAdminComm','StudyManagingRDO','StudyManagingSpecialty','StudyLeadLCRN','StudyPriorityPandemic','StudyPortfolioEligibility','StudyGovernanceRoute','StudyManagingSpecialty_PrimarySubSpecialty','StudyRandomisationStatus','StudyDesignType','StudyIntervention','StudyDesignTypeIntOb','StudyGeographicalScope','StudyCommercialInvolvement','StudySampleSizeUK','StudySampleSizeGlobal','StudySampleSizeNI','StudyManagingRDOShort','StudySampleSizeWales','StudySampleSizeEngland','StudyHasScreeningElement','StudyShouldUploadRecruitmentData','StudyHasMainFunder','StudySponsorType','StudyConsumerInvolvement','StudyConsumerInvolvementDetail','Study_IsUrgentPublicHealthResearch','Study_IsExperimentalMedicineComponent','Study_IsCommercial','Study_IsCTU','StudyCountDraftAndLive','Study_IsPostApril2010','Study_IsInPublicSearch','Study_IsHRCSICDReviewed','StudyComplexityCategory','StudyCommercialOrCollaborative','StudyCommercialStudy','StudyCount','StudyResearchCategory','StudyCommercialStudyType','StudyFeasibilityStatus','StudyMDSComplete','Study_IsJDRStudy','StudySampleSizeScotland','Study_IsCROStudy','StudyResearchAssessment','StudyComplexityCategoryExtra','StudyPriority','Study_IsNonNHS','StudyComplexityCategoryWeighting','StudyLeadLCRNAcronym','StudyComplexityCategoryLargeOther','StudyUpperAgeLimit','StudyLowerAgeLimit','Study_PandemicStatusConfirmed','Study_IsManagedRecovery','Study_IsCOVID','Study_DoesStudyRecordParticipantAge']]

#split into X (inputs-features) and y(output variable):
X = data.drop('ICDBlock', axis=1)
y=data['ICDBlock']

#split data into categorical and numerical
#get list of categorical columns: (includes categorical where it is binary -1 & 0)
#ohe_col = data.select_dtypes(include=['object']).columns.tolist()
all_col = X.columns.tolist() #get list of all columns 
#num_col = all_col #instantiate a list of numerical columns 
num_col = X._get_numeric_data().columns
ohe_col = list(set(all_col) - set(num_col)) #getting list of one-hot enoding columns - categorical features
#loop to get list of remainder columns after excluding ohe columns:
# for a in all_col:
#   if a in ohe_col:
#     num_col.remove(a)
X_ohe = X.drop(columns = ohe_col,inplace=False) #dataset for one-hot encoding transformation
X_num = X.drop(columns = num_col,inplace=False) #dataset for with numerical columns 

#one-hot encoding
enc = preprocessing.OneHotEncoder(handle_unknown='ignore')

# 2. FIT
enc.fit(X_ohe)

# 3. Transform
onehotlabels = enc.transform(X_ohe).toarray()
onehotlabels.shape

#remove NAs or blanks
data = data.fillna(0, inplace=True)
#convert to float
data = data.astype(float)

# Convert to NumPy as required for k-fold splits
X_np = X.values
y_np = y.values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size = 0.25, random_state=42)

# Scale X data
X_train_sc, X_test_sc = pr.scale_data(X_train, X_test)


#load model
model = nn.make_net(10)
model.summary() #look at model summary - arbitrarily 


# Define network
number_features = X_train_sc.shape[1]
model = nn.make_net(number_features)

### Train model (and store training info in history)
history = model.fit(X_train_sc,
                    y_train,
                    epochs=250,
                    batch_size=64,
                    validation_data=(X_test_sc, y_test),
                    verbose=1)

# Show acuracy
nn.calculate_accuracy(model, X_train_sc, X_test_sc, y_train, y_test)