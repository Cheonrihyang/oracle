const deleteButton= document.getElementById("delete-btn");
if(deleteButton){
    deleteButton.addEventListener('click',event =>{
        let id = document.getElementById("article-id").value;

        fetch(`/api/article/${id}`,{
            method:'DELETE'
        })
            .then(()=>{
                alert("삭제가 완료되었습니다.");
                location.replace('/articles')
            });
    })
}

const modifyButton= document.getElementById("modify-btn");
if(modifyButton){
    modifyButton.addEventListener('click',event =>{
        let params = new URLSearchParams(location.search);
        let id = params.get("id");

        fetch(`/api/article/${id}`,{
            method:'PUT',
            headers:{
                "Content-Type":"application/json",
            },
            body:JSON.stringify({
                title:document.getElementById('title').value,
                content:document.getElementById('content').value
            })
        })
            .then(()=>{
                alert("수정이 완료되었습니다.");
                location.replace('/articles')
            });
    })
}

const insertButton= document.getElementById("insert-btn");
if(insertButton){
    insertButton.addEventListener('click',event =>{

        fetch(`/api/article`,{
            method:'GET',
            headers:{
                "Content-Type":"application/json",
            },
            body:JSON.stringify({
                title:document.getElementById('title').value,
                content:document.getElementById('content').value
            })
        })
            .then(()=>{
                alert("입력이 완료되었습니다.");
                location.replace('/articles')
            });
    })
}