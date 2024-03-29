# Datasheet: Indoor 3D Egocentric Object (IDEO)

Based on the template from Gebru, Timnit, et al. "Datasheets for datasets." Communications of the ACM 64.12 (2021): 86-92.

## Motivation

*The questions in this section are primarily intended to encourage dataset creators to clearly articulate their reasons for creating the dataset and to promote transparency about funding interests.*

1. **For what purpose was the dataset created?** Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.

	The dataset was created to enable research on egocentric 3D object perception with a focus on category-level object pose/shape and scale prediction from single view RGB(-D) image. 

2. **Who created this dataset (e.g. which team, research group) and on behalf of which entity (e.g. company, institution, organization)**?

	The dataset was created by Tien Do, Lance Lemke, Jingfan Guo, Khiem Vuong, Minh Vo, and Hyun Soo Park in affiliation with University of Minnesota, Carnegie Mellon University, and Meta Reality Labs.

3. **What support was needed to make this dataset?** (e.g. who funded the creation of the dataset? If there is an associated grant, provide the name of the grantor and the grant name and number, or if it was supported by a company or government agency, give those details.)

	Funding was provided by the National Science Foundation.

4. **Any other comments?**

	None.


## Composition

*Dataset creators should read through the questions in this section prior to any data collection and then provide answers once collection is complete. Most of these questions are intended to provide dataset consumers with the information they need to make informed decisions about using the dataset for specific tasks. The answers to some of these questions reveal information about compliance with the EU’s General Data Protection Regulation (GDPR) or comparable regulations in other jurisdictions.*

1. **What do the instances that comprise the dataset represent (e.g. documents, photos, people, countries)?** Are there multiple types of instances (e.g. movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.

	The instances are videos taken by people wearing a head-mounted Kinect Azure cameras performing diverse daily indoor activities, e.g., cleaning, cooking, shopping, dish washing, and vacuuming, etc.

2. **How many instances are there in total (of each type, if appropriate)?**

	There are 58K RGB-D images with multiple 3D fitted object models, selected from 100 hours of egocentric videos performed by 85 subjects.

3. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g. geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g. to cover a more diverse range of instances, because instances were withheld or unavailable).

	The dataset is a sample of instances. It is intended to be a random sample of recorded daily activities from a diverse group of participants. No tests were run to determine representativeness. 

4. **What data does each instance consist of?** "Raw" data (e.g. unprocessed text or images) or features? In either case, please provide a description.

	Each raw data point consists of synchronized RGB-D image, where each object in the image is associated with their corresponding mesh, category label, 2D instance segmentation masks, and annotation data including its scale, orientation, and translation with respect to the camera.

5. **Is there a label or target associated with each instance?** If so, please provide a description.

	See above.

6. **Is any information missing from individual instances?** If so, please provide a description, explaining why this information is missing (e.g. because it was unavailable). This does not include intentionally removed information, but might include, e.g. redacted text.

	Everything is included. No data is missing.

7. **Are relationships between individual instances made explicit (e.g. users' movie ratings, social network links)?** If so, please describe how these relationships are made explicit.

	None explicitly.

8. **Are there recommended data splits (e.g. training, development/validation, testing)?** If so, please provide a description of these splits, explaining the rationale behind them.

	There are recommended data splits, see more in our Supplementary Material. We tried our best to make sure the testing/validation set is diverse and there's no overlapping between data in the training and testing/validation splits (e.g., different participants/scenes in different splits).

9. **Are there any errors, sources of noise, or redundancies in the dataset?** If so, please provide a description.

	See Preprocessing below.

10. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g. websites, tweets, other datasets)?** If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g. licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.

	The dataset is entirely self-contained.

11. **Does the dataset contain data that might be considered confidential (e.g. data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)?** If so, please provide a description.

	Unknown to the authors of the dataset.

12. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** If so, please describe why.

	Unknown to the authors of the dataset.

13. **Does the dataset relate to people?** If not, you may skip the remaining questions in this section.

	Yes.

14. **Does the dataset identify any subpopulations (e.g. by age, gender)?** If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.

	No.

15. **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** If so, please describe how.

	No. Participants' identifications are not revealed. Sensitive data such as their faces and personal information will be redacted. 

16. **Does the dataset contain data that might be considered sensitive in any way (e.g. data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** If so, please provide a description.

	Sensitive data such as participants' faces and their personal information will be redacted. 

17. **Any other comments?**

	None.


## Collection

*As with the previous section, dataset creators should read through these questions prior to any data collection to flag potential issues and then provide answers once collection is complete. In addition to the goals of the prior section, the answers to questions here may provide information that allow others to reconstruct the dataset without access to it.*

1. **How was the data associated with each instance acquired?** Was the data directly observable (e.g. raw text, movie ratings), reported by subjects (e.g. survey responses), or indirectly inferred/derived from other data (e.g. part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.

	The data was mostly observable as raw RGB-D images/videos, except that there are additional labels/annotations being extracted by the process described in the main paper.

2. **What mechanisms or procedures were used to collect the data (e.g. hardware apparatus or sensor, manual human curation, software program, software API)?** How were these mechanisms or procedures validated?

	The subjects are asked to wear a head-mounted Kinect Azure camera while performing diverse daily indoor activities, e.g., cleaning, cooking, shopping, dish washing, and vacuuming.

3. **If the dataset is a sample from a larger set, what was the sampling strategy (e.g. deterministic, probabilistic with specific sampling probabilities)?**

	We randomly selected a subset of data that comprises diverse daily activities.

4. **Who was involved in the data collection process (e.g. students, crowdworkers, contractors) and how were they compensated (e.g. how much were crowdworkers paid)?**

	We recruited random participants to collect data. The participants received 35 USD for one hour of data.

5. **Over what timeframe was the data collected?** Does this timeframe match the creation timeframe of the data associated with the instances (e.g. recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created. Finally, list when the dataset was first published.

	The data was collected from May 2021 to Oct 2021.

7. **Were any ethical review processes conducted (e.g. by an institutional review board)?** If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.

	Yes. We provide extra information at our recruiting website https://z.umn.edu/ideadc where the Institutional Review Board (IRB) approved consent form is available for the participant to sign before taking the data. We also attach this form (consent_form.pdf) with the supplementary material for review.

8. **Does the dataset relate to people?** If not, you may skip the remainder of the questions in this section.

	Yes.

9. **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g. websites)?**

	We collected the data from the individuals in question directly.

10. **Were the individuals in question notified about the data collection?** If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.

	Yes. We put up a recruiting flyer for the participants to sign up (see recruiting_flyer.pdf in Supplementary). The data collection kit was delivered to their residences with clear instructions on how to start/stop recording data. 

11. **Did the individuals in question consent to the collection and use of their data?** If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.

	Yes. All the participants signed the Institutional Review Board (IRB) approved consent form before taking the data (see consent_form.pdf in Supplementary).

12. **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).

	The participants can contact us directly in case they want to revoke their consent in the future.

13. **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g. a data protection impact analysis) been conducted?** If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.

	N/A.

14. **Any other comments?**

	None.


## Preprocessing / Cleaning / Labeling

*Dataset creators should read through these questions prior to any pre-processing, cleaning, or labeling and then provide answers once these tasks are complete. The questions in this section are intended to provide dataset consumers with the information they need to determine whether the “raw” data has been processed in ways that are compatible with their chosen tasks. For example, text that has been converted into a “bag-of-words” is not suitable for tasks involving word order.*

1. **Was any preprocessing/cleaning/labeling of the data done (e.g. discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** If so, please provide a description. If not, you may skip the remainder of the questions in this section.

    Yes. We ask the trained users to filter out the corrupted data and identify the highly valuable parts of the raw data (e.g., we focus on hand-object interaction scenes), then we use Amazon Mechanical Turk crowd workers for annotating the 3D objects' 7-DoF (scale, rotation, and translation) manipulated by hand. Please see Section 3.2 in the main paper for more details.

2. **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g. to support unanticipated future uses)?** If so, please provide a link or other access point to the "raw" data.

    Yes. The raw data was saved, and we will release the entire raw data at the time of publication.

3. **Is the software used to preprocess/clean/label the instances available?** If so, please provide a link or other access point.

	We will release our annotation tool for public use.

4. **Any other comments?**

	None.


## Uses

*These questions are intended to encourage dataset creators to reflect on the tasks  for  which  the  dataset  should  and  should  not  be  used.  By  explicitly highlighting these tasks, dataset creators can help dataset consumers to make informed decisions, thereby avoiding potential risks or harms.*

1. **Has the dataset been used for any tasks already?** If so, please provide a description.

	At the time of publication, only the original paper.

2. **Is there a repository that links to any or all papers or systems that use the dataset?** If so, please provide a link or other access point.

	None at the time of publication.

3. **What (other) tasks could the dataset be used for?**

	The dataset could be used for any task related to 3D understanding of static and dynamic objects, especially for egocentric activities where dynamic objects are being manipulated by hands. 

4. **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g. stereotyping, quality of service issues) or other undesirable harms (e.g. financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?

	There is minimal risk for harm: we commit to redacting sensitive information that can be potentially used to identify the participants.

5. **Are there tasks for which the dataset should not be used?** If so, please provide a description.

	Systems trained on this data may or may not generalize well to other tasks if there exists a large domain gap between the testing and training data. Consequently, such systems should not - without additional verification - be used to make consequential decisions about people.  

6. **Any other comments?**

	None.


## Distribution

*Dataset creators should provide answers to these questions prior to distributing the dataset either internally within the entity on behalf of which the dataset was created or externally to third parties.*

1. **Will the dataset be distributed to third parties outside of the entity (e.g. company, institution, organization) on behalf of which the dataset was created?** If so, please provide a description.

	Yes, the dataset is publicly available on the internet

2. **How will the dataset will be distributed (e.g. tarball on website, API, GitHub)?** Does the dataset have a digital object identifier (DOI)?

	The dataset itself is hosted on an AWS S3 bucket. It is distributed on our GitHub repository: https://github.com/ideo-benchmark-challenges/IDEO

3. **When will the dataset be distributed?**

	It is currently available to download and use.

4. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.

	No.

5. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.

	No.

6. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.

	Unknown to authors of the datasheet.

7. **Any other comments?**

	None.


## Maintenance

*As with the previous section, dataset creators should provide answers to these questions prior to distributing the dataset. These questions are intended to encourage dataset creators to plan for dataset maintenance and communicate this plan with dataset consumers.*

1. **Who is supporting/hosting/maintaining the dataset?**

	The authors are supporting/maintaining the dataset.

2. **How can the owner/curator/manager of the dataset be contacted (e.g. email address)?**

	The first author of the dataset, Tien Do, can be contacted at doxxx104@umn.edu.

3. **Is there an erratum?** If so, please provide a link or other access point.

	We will maintain a list of known errors on the dataset GitHub repo.

4. **Will the dataset be updated (e.g. to correct labeling errors, add new instances, delete instances)?** If so, please describe how often, by whom, and how updates will be communicated to users (e.g. mailing list, GitHub)?

	Yes. Subsequent corrections/updates will be posted on the dataset GitHub repo.

5. **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g. were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** If so, please describe these limits and explain how they will be enforced.

	N/A.

6. **Will older versions of the dataset continue to be supported/hosted/maintained?** If so, please describe how. If not, please describe how its obsolescence will be communicated to users.

	Older versions will be kept around for consistency and comparison purposes.

7. **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.

	Others may do so and should contact the original authors about incorporating fixes/extensions.

8. **Any other comments?**

	None.
