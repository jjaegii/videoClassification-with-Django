from xml.etree.ElementPath import prepare_predicate
from . import prediction
from . import models
from django.shortcuts import get_object_or_404, render

# Create your views here.
def uploadFile(request):
    if request.method == "POST":
        # Fetching the form data
        fileTitle = request.POST["fileTitle"]
        uploadedFile = request.FILES["uploadedFile"]
        print("fileTitle:" + fileTitle)
        print("uploadedFileName:" + uploadedFile.name)
        # Saving the information in the database
        document = models.Document(
            title = fileTitle,
            uploadedFile = uploadedFile,
            predictResult = "Error"
            #predictResult = predictResult
        )
        document.save()
        document.predictResult = prediction.sequence_prediction(uploadedFile.name) 
        document.uploadedFile = "Uploaded Files/" + uploadedFile.name.replace(".mp4", ".gif")
        document.save()
        print("prediction:" + document.predictResult)
        

    documents = models.Document.objects.all()

    return render(request, "Core/upload-file.html", context = {
        "files": documents
    })
